import fs from 'node:fs';
import solc from 'solc';
const source=fs.readFileSync('contracts/TrustLedgerV4.sol','utf8');
const input={language:'Solidity',sources:{'TrustLedgerV4.sol':{content:source}},settings:{optimizer:{enabled:true,runs:200},evmVersion:'cancun',outputSelection:{'*':{'*':['abi','evm.bytecode.object']}}}};
const result=JSON.parse(solc.compile(JSON.stringify(input)));
for(const error of result.errors??[]) console.error(error.formattedMessage);
if((result.errors??[]).some(x=>x.severity==='error')) process.exit(1);
fs.mkdirSync('build',{recursive:true});
fs.writeFileSync('build/TrustLedgerV4.json',JSON.stringify({...result.contracts['TrustLedgerV4.sol'].TrustLedgerV4,compiler:solc.version(),input},null,2));
