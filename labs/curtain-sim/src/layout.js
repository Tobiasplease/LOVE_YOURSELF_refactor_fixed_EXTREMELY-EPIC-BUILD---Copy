import { Bridle, Motor } from './rig.js';

export const ROW_KINDS = {
  top: { label: 'TOP', depth: 0.26, rods: true },
  belly: { label: 'BELLY', depth: 0.58, rods: true },
  hem: { label: 'HEM', depth: 1, rods: false },
};

export function rowKinds(rowCount) {
  return rowCount >= 3 ? ['top', 'belly', 'hem'] : ['belly', 'hem'];
}

export function buildRig(curtain, p, ceilingY) {
  const kinds = rowKinds(p.rows);
  const pitch = curtain.fabricWidth / p.columns;
  const spread = Math.min(0.16, pitch * 0.27);
  const rodLength = Math.min(0.2, pitch * 0.45);
  const railLeft = curtain.o.x - curtain.o.railWidth / 2;
  const lines = [];
  const rods = [];

  kinds.forEach((kind, rowIndex) => {
    const info = ROW_KINDS[kind];
    const row = Math.min(curtain.ny - 1, Math.round((curtain.ny - 1) * info.depth));
    for (let c = 0; c < p.columns; c++) {
      const u = pitch * (c + 0.5);
      const cols = [u - spread, u + spread].map((v) => Math.max(0, Math.min(curtain.nx - 1, Math.round(v / curtain.dx))));
      if (p.rods && info.rods) for (const col of cols) rods.push(curtain.addRod(row, col, rodLength, 0.0012));

      const railX = railLeft + (curtain.o.railWidth * (c + 0.5)) / p.columns;
      const motor = new Motor({ maxSpeed: p.motorSpeed });
      const bridle = new Bridle(curtain, {
        attach: cols.map((col) => curtain.index(col, row)),
        anchor: [railX, ceilingY - 0.03, curtain.o.z + p[`${kind}Dist`]],
        motor,
        elasticK: p.elasticK,
        reach: p.legReach,
      });
      lines.push({ kind, rowIndex, column: c, motor, bridle, railX, along: (c + 0.5) / p.columns, below: info.depth * curtain.o.height });
    }
  });
  return { lines, rods, kinds };
}
