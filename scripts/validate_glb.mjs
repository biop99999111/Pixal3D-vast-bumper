import fs from 'node:fs/promises';
import path from 'node:path';
import validator from 'gltf-validator';

const [asset, output] = process.argv.slice(2);
if (!asset || !output) {
  console.error('Usage: node scripts/validate_glb.mjs asset.glb report.json');
  process.exit(2);
}
const bytes = new Uint8Array(await fs.readFile(asset));
const report = await validator.validateBytes(bytes, {uri: path.basename(asset)});
await fs.mkdir(path.dirname(output), {recursive: true});
await fs.writeFile(output, JSON.stringify(report, null, 2));
console.log(`glTF validation: ${report.issues.numErrors} errors, ${report.issues.numWarnings} warnings`);
process.exitCode = report.issues.numErrors ? 1 : 0;
