const el = (tag, cls, text) => {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text != null) node.textContent = text;
  return node;
};

export function buildGroup(root, title, items, params, onChange) {
  const box = el('fieldset', 'grp');
  box.append(el('legend', 'cap', title));
  const body = el('div', 'faders');
  box.append(body);
  const refreshers = [];

  for (const item of items) {
    if (item.type === 'choice') {
      const wrap = el('div', 'choice');
      wrap.append(el('span', 'lbl', item.label));
      const row = el('div', 'seg');
      const buttons = item.options.map(([value, label]) => {
        const b = el('button', 'mini', label);
        b.addEventListener('click', () => {
          params[item.key] = value;
          refresh();
          onChange(item);
        });
        row.append(b);
        return [value, b];
      });
      const refresh = () => buttons.forEach(([v, b]) => b.classList.toggle('on', params[item.key] === v));
      refresh();
      refreshers.push(refresh);
      wrap.append(row);
      body.append(wrap);
      continue;
    }

    if (item.type === 'action') {
      const b = el('button', 'mini', item.label);
      b.addEventListener('click', () => {
        item.run();
        onChange(item);
      });
      body.append(b);
      continue;
    }

    if (item.type === 'toggle') {
      const b = el('button', 'mini toggle', item.label);
      const refresh = () => b.classList.toggle('on', !!params[item.key]);
      b.addEventListener('click', () => {
        params[item.key] = !params[item.key];
        refresh();
        onChange(item);
      });
      refresh();
      refreshers.push(refresh);
      body.append(b);
      continue;
    }

    const wrap = el('label', 'fader');
    const head = el('span', 'head');
    const value = el('b');
    head.append(el('span', 'nm', item.label), value);
    const input = el('input');
    input.type = 'range';
    input.id = `f-${item.key}`;
    Object.assign(input, { min: item.min, max: item.max, step: item.step ?? 0.01 });
    const show = () => {
      input.value = params[item.key];
      value.textContent = `${Number(params[item.key]).toFixed(item.digits ?? 2)}${item.unit ?? ''}`;
    };
    input.addEventListener('input', () => {
      params[item.key] = Number(input.value);
      show();
      if (!item.rebuild) onChange(item);
    });
    input.addEventListener('change', () => item.rebuild && onChange(item));
    show();
    refreshers.push(show);
    wrap.append(head, input);
    body.append(wrap);
  }

  root.append(box);
  return () => refreshers.forEach((r) => r());
}

export function buildMatrix(root, lines, kinds, columns) {
  root.innerHTML = '';
  root.style.gridTemplateColumns = `auto repeat(${columns}, minmax(0, 1fr))`;
  const cells = [];
  kinds.forEach((kind, r) => {
    root.append(el('span', 'rowname', kind.toUpperCase()));
    for (let c = 0; c < columns; c++) {
      const line = lines.find((l) => l.rowIndex === r && l.column === c);
      const cell = el('div', 'cell');
      const fill = el('i');
      const text = el('span');
      cell.append(fill, text);
      root.append(cell);
      cells.push({ line, cell, fill, text, peak: 0 });
    }
  });
  return {
    update(maxPull) {
      for (const c of cells) {
        const { motor, bridle } = c.line;
        const grams = (bridle.tension / 9.81) * 1000;
        c.peak = Math.max(c.peak, grams);
        c.fill.style.height = `${Math.min(100, (motor.pos / Math.max(0.01, maxPull)) * 100)}%`;
        c.cell.classList.toggle('limit', motor.saturated);
        c.text.textContent = `${Math.round(motor.pos * 100)}cm\n${Math.round(grams)}g`;
      }
    },
    peaks() {
      return cells.map((c) => c.peak);
    },
    resetPeaks() {
      for (const c of cells) c.peak = 0;
    },
  };
}
