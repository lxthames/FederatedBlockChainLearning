import fs from 'node:fs';
import solc from 'solc';
const source = fs.readFileSync('contracts/TrustLedger.sol','utf8');
const input = {language:'Solidity', sources:{'TrustLedger.sol':{content:source}},
  settings:{optimizer:{enabled:true,runs:200},evmVersion:'cancun',
    outputSelection:{'*':{'*':['abi','evm.bytecode.object']}}}};
const result = JSON.parse(solc.compile(JSON.stringify(input)));
for (const error of result.errors ?? []) console.error(error.formattedMessage);
if ((result.errors ?? []).some(x=>x.severity==='error')) process.exit(1);
fs.mkdirSync('build',{recursive:true});
fs.writeFileSync('build/TrustLedger.json',JSON.stringify({
  ...result.contracts['TrustLedger.sol'].TrustLedger,compiler:solc.version(),input},null,2));
