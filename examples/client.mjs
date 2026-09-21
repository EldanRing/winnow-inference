// Node 20+, no runtime dependencies. Calls the typed-decision HTTP endpoint.
import {readFile} from 'node:fs/promises';
const request=JSON.parse(await readFile(new URL('./decisions.json',import.meta.url),'utf8'));
const response=await fetch('http://127.0.0.1:8091/v1/systemone',{
  method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(request)
});
if(!response.ok)throw new Error(await response.text());
console.log(await response.json());
