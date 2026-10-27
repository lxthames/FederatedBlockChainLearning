// SPDX-License-Identifier: MIT
pragma solidity ^0.8.30;

/// @notice Experimental model-state ledger. Reports attest to measured quality;
/// they do not prove the cause of a degradation. The registrar is trusted.
contract TrustLedgerV4 {
    address public immutable registrar;
    bytes32 public immutable experiment;
    uint256 public immutable minimumStake;
    uint256 public immutable penalty;
    uint256 public latestVersion;
    uint256 public activeVersion;
    uint256 public recoveryCount;
    uint256 public burned;
    mapping(address => bool) public enrolled;
    mapping(address => uint256) public stake;
    mapping(uint256 => bytes32) public modelHash;
    mapping(uint256 => uint256) public parent;
    mapping(uint256 => bool) public invalidated;
    mapping(uint256 => address[]) private modelSigners;
    mapping(uint256 => mapping(address => bool)) private signedModel;
    struct Candidate {
        uint256 round;
        uint256 baseVersion;
        address proposer;
        bytes32 updateHash;
        uint256 quorum;
        uint256 approvals;
        bool exists;
        address[] members;
        address[] affirmative;
    }
    mapping(bytes32 => Candidate) private candidates;
    mapping(bytes32 => mapping(address => bool)) private member;
    mapping(bytes32 => mapping(address => bool)) private voted;
    struct Recovery {
        uint256 first;
        uint256 last;
        uint256 restored;
        uint256 quorum;
        uint256 approvals;
        bool exists;
        bool applied;
        bool penaltiesEnabled;
        bytes32 reason;
    }
    mapping(bytes32 => Recovery) public recoveries;
    mapping(bytes32 => mapping(address => bool)) private recoveryMember;
    mapping(bytes32 => mapping(address => bool)) private recoveryVoted;
    mapping(bytes32 => mapping(address => bool)) private penalized;
    event Enrolled(address indexed account);
    event Deposited(address indexed account, uint256 amount);
    event CandidateOpened(bytes32 indexed key, uint256 round, address proposer, bytes32 updateHash);
    event Vote(bytes32 indexed key, address indexed validator, bool approve);
    event ModelCommitted(uint256 indexed version, uint256 parentVersion, bytes32 hash);
    event RecoveryApplied(bytes32 indexed key, uint256 first, uint256 last, uint256 restored);
    event Slashed(bytes32 indexed recovery, address indexed validator, uint256 amount);
    modifier onlyRegistrar() { require(msg.sender == registrar, 'registrar'); _; }
    constructor(bytes32 id, uint256 minimum, uint256 amount) {
        require(minimum > 0 && amount > 0, 'parameters');
        registrar = msg.sender; experiment = id; minimumStake = minimum; penalty = amount;
    }
    function enroll(address account) external onlyRegistrar {
        require(!enrolled[account] && account != address(0), 'enrollment');
        enrolled[account] = true; emit Enrolled(account);
    }
    function deposit() external payable {
        require(enrolled[msg.sender] && msg.value > 0, 'deposit');
        stake[msg.sender] += msg.value; emit Deposited(msg.sender, msg.value);
    }
    function genesis(bytes32 hash) external onlyRegistrar {
        require(modelHash[0] == bytes32(0) && hash != bytes32(0), 'genesis');
        modelHash[0] = hash; emit ModelCommitted(0,0,hash);
    }
    function candidateKey(uint256 round, uint256 base, address proposer, bytes32 hash) public view returns(bytes32) {
        return keccak256(abi.encode(block.chainid,address(this),experiment,round,base,proposer,hash));
    }
    function openCandidate(uint256 round, address proposer, bytes32 hash, address[] calldata members, address[] calldata eligible) external onlyRegistrar returns(bytes32 key) {
        require(modelHash[0] != bytes32(0) && round == latestVersion+1, 'round');
        require(enrolled[proposer] && stake[proposer]>=minimumStake && hash != bytes32(0), 'candidate');
        // Registrar attests off-chain trust eligibility. Contract checks the
        // declared full snapshot for stake, uniqueness, exclusion and size.
        uint256 count=eligible.length;
        require(count>=3,'eligible population');
        uint256 size=(3*count+9)/10;
        if(size<3) size=3;
        if(size>count) size=count;
        require(members.length==size,'committee size');
        for(uint256 i=0;i<count;i++) {
            require(eligible[i]!=proposer && enrolled[eligible[i]] && stake[eligible[i]]>=minimumStake,'eligible');
            for(uint256 j=0;j<i;j++) require(eligible[i]!=eligible[j],'duplicate eligible');
        }
        key = candidateKey(round,activeVersion,proposer,hash);
        Candidate storage c = candidates[key]; require(!c.exists,'duplicate candidate');
        c.round=round; c.baseVersion=activeVersion; c.proposer=proposer; c.updateHash=hash;
        c.quorum=(2*members.length+2)/3; c.exists=true;
        for(uint256 i=0;i<members.length;i++) {
            address a=members[i];
            bool included=false;
            for(uint256 j=0;j<count;j++) if(eligible[j]==a) included=true;
            require(included,'outside snapshot');
            require(a!=proposer && enrolled[a] && stake[a]>=minimumStake && !member[key][a], 'member');
            member[key][a]=true; c.members.push(a);
        }
        emit CandidateOpened(key,round,proposer,hash);
    }
    function vote(bytes32 key, bool approve) external {
        Candidate storage c=candidates[key];
        require(c.exists && c.round==latestVersion+1 && c.baseVersion==activeVersion,'stale vote');
        require(member[key][msg.sender] && !voted[key][msg.sender],'vote');
        voted[key][msg.sender]=true;
        if(approve) { c.approvals++; c.affirmative.push(msg.sender); }
        emit Vote(key,msg.sender,approve);
    }
    function accepted(bytes32 key) external view returns(bool) {
        Candidate storage c=candidates[key]; return c.exists && c.approvals>=c.quorum;
    }
    function commitModel(uint256 version, bytes32 hash, bytes32[] calldata approved) external onlyRegistrar {
        require(modelHash[0]!=bytes32(0) && version==latestVersion+1 && hash!=bytes32(0),'version');
        for(uint256 i=0;i<approved.length;i++) {
            Candidate storage c=candidates[approved[i]];
            require(c.exists && c.round==version && c.baseVersion==activeVersion && c.approvals>=c.quorum,'quorum');
            for(uint256 j=0;j<c.affirmative.length;j++) {
                address a=c.affirmative[j];
                if(!signedModel[version][a]) { signedModel[version][a]=true; modelSigners[version].push(a); }
            }
        }
        // The trusted orchestrator computes aggregation; an empty approved set
        // records a no-op model version, not an implicitly approved update.
        parent[version]=activeVersion; modelHash[version]=hash;
        emit ModelCommitted(version,activeVersion,hash);
        latestVersion=version; activeVersion=version;
    }
    function openRecovery(bytes32 key, uint256 first, uint256 last, uint256 restored,
        bytes32 reason, address[] calldata panel, bool penaltiesEnabled) external onlyRegistrar {
        require(!recoveries[key].exists && first>0 && first<=last && last==latestVersion && activeVersion==last,'interval');
        require(restored<first && modelHash[restored]!=bytes32(0) && !invalidated[restored] && panel.length>=3,'restore');
        Recovery storage r=recoveries[key];
        r.first=first; r.last=last; r.restored=restored; r.quorum=(2*panel.length+2)/3;
        r.exists=true; r.reason=reason; r.penaltiesEnabled=penaltiesEnabled;
        for(uint256 i=0;i<panel.length;i++) {
            address a=panel[i];
            require(enrolled[a] && stake[a]>=minimumStake && !recoveryMember[key][a],'panel');
            recoveryMember[key][a]=true;
        }
    }
    function attestRecovery(bytes32 key) external {
        Recovery storage r=recoveries[key];
        require(r.exists && !r.applied && activeVersion==r.last && recoveryMember[key][msg.sender] && !recoveryVoted[key][msg.sender],'attestation');
        recoveryVoted[key][msg.sender]=true; r.approvals++;
    }
    function applyRecovery(bytes32 key) external onlyRegistrar {
        Recovery storage r=recoveries[key];
        require(r.exists && !r.applied && r.approvals>=r.quorum && activeVersion==r.last,'recovery quorum');
        r.applied=true; recoveryCount++;
        for(uint256 v=r.first;v<=r.last;v++) {
            invalidated[v]=true;
            if(r.penaltiesEnabled) for(uint256 j=0;j<modelSigners[v].length;j++) {
                address a=modelSigners[v][j];
                if(!penalized[key][a]) {
                    penalized[key][a]=true;
                    uint256 amount=stake[a]<penalty ? stake[a] : penalty;
                    stake[a]-=amount; burned+=amount; emit Slashed(key,a,amount);
                }
            }
        }
        activeVersion=r.restored;
        emit RecoveryApplied(key,r.first,r.last,r.restored);
    }
}
