"""Explicit, auditable insertions into the frozen V17 runner; never edits V17."""
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'scripts/trust_alignment_v17/runner.py'


def rendered_runner():
    value=SOURCE.read_text(encoding='utf-8')
    edits=[
        ('from .behavior import AlignmentRound as ChallengeRound',
         'from scripts.trust_alignment_v17.behavior import AlignmentRound as ChallengeRound\nfrom scripts.trust_supervised_v18.repair import checked_load'),
        ("        self.challenge_arm = 'current'", "        self.online_controller = None\n        self.offline_observer = None\n        self.challenge_arm = 'current'"),
        ("            self.logger.log('model_integrity',rnd,dict(parent_hash=state_hash(round_parent),", """            # The inherited aggregate is committed and checked above. The online
            # controller only sees repair/private validation data, never test data.
            aggregate_commit_hash = committed_hash
            online_validated_hash = None
            if self.online_controller is not None and self.online_controller.due(rnd):
                outcome = self.online_controller.attempt(model,x,y,split,rnd,participants,c.seed)
                # Offline measurements happen after the final admission decision.
                # Their return values never feed repair, admission or scheduling.
                if self.offline_observer is not None:
                    self.offline_observer(outcome,rnd)
                assert_hash(model,aggregate_commit_hash,'offline observer preserves aggregate')
                assert_hash(outcome.candidate,outcome.payload['candidate_model_hash'],'offline observer preserves candidate')
                assert_hash(outcome.committed,outcome.payload['committed_model_hash'],'offline observer preserves selection')
                expected_commit = snapshot(outcome.committed)
                checked_load(model,expected_commit,'online repair commit')
                committed_hash = assert_state(model,expected_commit,'online committed model equality')
                online_validated_hash = outcome.payload['decision']['scored_model_hash'] if outcome.payload['decision']['passed'] else aggregate_commit_hash
                if committed_hash != online_validated_hash:
                    raise IntegrityError('online committed model differs from validated selection')
                transmitted += outcome.payload['extra_transmitted_bytes']
                self.online_controller.persist(outcome,self.output_dir,rnd)
                self.logger.log('online_repair',rnd,outcome.payload)
            self.logger.log('model_integrity',rnd,dict(parent_hash=state_hash(round_parent),
                aggregate_committed_hash=aggregate_commit_hash,online_validated_hash=online_validated_hash,"""),
        ("        (self.output_dir/'chain-trace.json').write_text", "        self.final_model = model\n        (self.output_dir/'chain-trace.json').write_text"),
    ]
    for old,new in edits:
        if value.count(old)!=1:raise ValueError(f'unexpected inherited insertion: {old}')
        value=value.replace(old,new)
    return value


if __name__=='__main__':
    (Path(__file__).parent/'runner.py').write_text(rendered_runner(),encoding='utf-8')
