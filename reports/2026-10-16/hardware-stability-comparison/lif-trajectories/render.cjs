// Standard SVG rasterization, no image generation or network requests.
const fs=require('fs');const path=require('path');const crypto=require('crypto');const sharp=require('sharp');
const sha=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
(async()=>{
  const base=__dirname,svg=path.join(base,'LIF_TRAJECTORIES.svg'),png=path.join(base,'LIF_TRAJECTORIES.png');
  const pp=path.join(base,'PLOT_PROVENANCE.json'),p=JSON.parse(fs.readFileSync(pp,'utf8'));
  if(sha(svg)!==p.outputs['LIF_TRAJECTORIES.svg'].sha256)throw Error('SVG hash changed');
  await sharp(svg,{density:144}).png({compressionLevel:9}).toFile(png);
  p.outputs['LIF_TRAJECTORIES.png']={bytes:fs.statSync(png).size,sha256:sha(png)};
  p.renderer={library:'sharp',version:sharp.versions.sharp,density:144,script_sha256:sha(__filename)};
  fs.writeFileSync(pp,JSON.stringify(p,null,2)+'\n');
  const total=fs.readdirSync(base).reduce((n,f)=>n+fs.statSync(path.join(base,f)).size,0);
  if(total>1000000)throw Error('Figure package exceeds 1 MB');
  process.stdout.write(JSON.stringify({png,total_bytes:total}));
})().catch(e=>{process.stderr.write(String(e));process.exitCode=1;});
