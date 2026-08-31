#!/usr/bin/env python3
"""
Real-time visualization of the activation memory network.
Run: python debug/activation_visualizer.py
Open: http://localhost:5050
"""

import json
import os
import sys
import time
from threading import Thread

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, jsonify, render_template_string

app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Activation Memory Visualizer</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {
            margin: 0;
            background: #1a1a2e;
            font-family: 'Segoe UI', sans-serif;
            color: #eee;
            overflow: hidden;
        }
        #container {
            display: flex;
            height: 100vh;
        }
        #graph {
            flex: 1;
            position: relative;
        }
        #sidebar {
            width: 320px;
            padding: 20px;
            background: #16213e;
            overflow-y: auto;
            border-left: 1px solid #0f3460;
        }
        h2 {
            color: #e94560;
            margin-top: 0;
            font-size: 1.2em;
        }
        h3 {
            color: #00d9ff;
            font-size: 1em;
            margin: 15px 0 10px 0;
        }
        .stat-row {
            display: flex;
            justify-content: space-between;
            padding: 5px 0;
            border-bottom: 1px solid #0f3460;
        }
        .stat-label { color: #888; }
        .stat-value { color: #fff; font-weight: bold; }
        .concept-list {
            max-height: 200px;
            overflow-y: auto;
        }
        .concept-item {
            display: flex;
            justify-content: space-between;
            padding: 4px 8px;
            margin: 2px 0;
            background: #0f3460;
            border-radius: 4px;
        }
        .concept-name { color: #fff; }
        .concept-value {
            color: #00d9ff;
            font-family: monospace;
        }
        .belief-item {
            padding: 8px;
            margin: 4px 0;
            background: #1a1a2e;
            border-left: 3px solid #e94560;
            font-style: italic;
            font-size: 0.9em;
        }
        .memory-item {
            padding: 8px;
            margin: 4px 0;
            background: #1a1a2e;
            border-left: 3px solid #00d9ff;
            font-size: 0.85em;
        }
        .memory-time {
            color: #666;
            font-size: 0.8em;
        }
        #status {
            position: fixed;
            top: 10px;
            left: 10px;
            padding: 8px 15px;
            background: rgba(0,0,0,0.7);
            border-radius: 20px;
            font-size: 0.9em;
        }
        .status-live { color: #00ff88; }
        .status-stale { color: #ff6b6b; }
        svg { width: 100%; height: 100%; }
        .node-label {
            font-size: 11px;
            fill: #fff;
            text-anchor: middle;
            pointer-events: none;
            text-shadow: 0 0 3px #000, 0 0 5px #000;
        }
        .legend {
            position: absolute;
            bottom: 20px;
            left: 20px;
            background: rgba(0,0,0,0.7);
            padding: 15px;
            border-radius: 8px;
        }
        .legend-item {
            display: flex;
            align-items: center;
            margin: 5px 0;
        }
        .legend-color {
            width: 20px;
            height: 20px;
            border-radius: 50%;
            margin-right: 10px;
        }
    </style>
</head>
<body>
    <div id="container">
        <div id="graph">
            <svg id="network"></svg>
            <div class="legend">
                <div class="legend-item">
                    <div class="legend-color" style="background: #ffd93d;"></div>
                    <span>Person / social</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #a8a8a8;"></div>
                    <span>Object / environment</span>
                </div>
                <div class="legend-item" style="font-size:0.8em; color:#666; margin-top:8px">
                    <span>Node size = activation level</span>
                </div>
            </div>
        </div>
        <div id="sidebar">
            <h2>Activation Memory</h2>

            <div class="stat-row">
                <span class="stat-label">Novelty:</span>
                <span class="stat-value" id="novelty">-</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Boredom:</span>
                <span class="stat-value" id="boredom">-</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Active Concepts:</span>
                <span class="stat-value" id="active-count">-</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Total Edges:</span>
                <span class="stat-value" id="edge-count">-</span>
            </div>

            <h3>Top Activated</h3>
            <div class="concept-list" id="top-concepts"></div>

            <h3>Learned Beliefs</h3>
            <div id="beliefs"></div>

            <h3>🧠 Compression State</h3>
            <div id="compression"></div>

            <h3>💭 Identity</h3>
            <div id="desires"></div>

            <h3>Recent Session Memories</h3>
            <div id="memories"></div>
        </div>
    </div>
    <div id="status">
        <span class="status-live">● LIVE</span>
    </div>

    <script>
        // Concept labels cache — populated from snapshot data
        let conceptLabels = {};

        // Person detection keywords for label-based categorization
        const PERSON_WORDS = ["person", "someone", "man", "woman", "people", "figure",
                              "sitting", "standing", "wearing", "seated", "individual",
                              "working", "typing", "looking", "human", "he ", "she "];

        function getConceptCategory(conceptId) {
            const label = (conceptLabels[conceptId] || '').toLowerCase();
            if (!label) return 'default';
            if (PERSON_WORDS.some(w => label.includes(w))) return 'social';
            return 'default';
        }

        function getConceptLabel(conceptId) {
            const label = conceptLabels[conceptId];
            if (!label) return conceptId;
            // Truncate long labels for display
            return label.length > 30 ? label.substring(0, 27) + '...' : label;
        }

        const width = document.getElementById('graph').clientWidth;
        const height = document.getElementById('graph').clientHeight;

        const svg = d3.select('#network')
            .attr('width', width)
            .attr('height', height);

        // Add zoom behavior
        const g = svg.append('g');
        svg.call(d3.zoom()
            .scaleExtent([0.3, 3])
            .on('zoom', (event) => g.attr('transform', event.transform)));

        function getCategoryColor(conceptId) {
            const cat = getConceptCategory(conceptId);
            switch(cat) {
                case 'social': return '#ffd93d';   // Yellow - person/engagement
                default: return '#a8a8a8';         // Gray - default
            }
        }

        // Size scale: activation -> node radius
        const sizeScale = d3.scaleLinear()
            .domain([0, 1])
            .range([8, 35]);

        let simulation = d3.forceSimulation()
            .force('link', d3.forceLink().id(d => d.id).distance(80))
            .force('charge', d3.forceManyBody().strength(-150))
            .force('center', d3.forceCenter(width / 2, height / 2))
            .force('collision', d3.forceCollide().radius(d => sizeScale(d.activation) + 5));

        let link = g.append('g').attr('class', 'links').selectAll('line');
        let node = g.append('g').attr('class', 'nodes').selectAll('circle');
        let label = g.append('g').attr('class', 'labels').selectAll('text');

        let lastUpdate = Date.now();
        let lastNodeCount = 0;
        let nodePositions = {};  // Preserve positions across updates

        function updateGraph(data) {
            // Update concept labels cache
            conceptLabels = data.concept_labels || {};

            // Build nodes from activations
            const newNodes = Object.entries(data.activations || {})
                .filter(([_, v]) => v > 0.05)
                .map(([id, activation]) => {
                    // Preserve existing position if we have it
                    const existing = nodePositions[id];
                    return {
                        id,
                        activation,
                        zone: data.spatial_tags?.[id] || 'unknown',
                        x: existing?.x,
                        y: existing?.y,
                        vx: existing?.vx || 0,
                        vy: existing?.vy || 0
                    };
                });

            // Build links from edges
            const links = [];
            const seen = new Set();
            for (const [source, targets] of Object.entries(data.edges || {})) {
                for (const [target, weight] of Object.entries(targets)) {
                    const key = [source, target].sort().join('-');
                    if (!seen.has(key) && weight > 0.1) {
                        seen.add(key);
                        links.push({ source, target, weight });
                    }
                }
            }

            // Only restart simulation if structure changed (new nodes added/removed)
            const structureChanged = newNodes.length !== lastNodeCount ||
                newNodes.some(n => !nodePositions[n.id]);
            lastNodeCount = newNodes.length;

            // Update simulation
            simulation.nodes(newNodes);
            simulation.force('link').links(links);

            if (structureChanged) {
                simulation.alpha(0.3).restart();
            }

            // Update links with transition
            link = link.data(links, d => `${d.source.id || d.source}-${d.target.id || d.target}`);
            link.exit().transition().duration(300).attr('stroke-opacity', 0).remove();
            link = link.enter()
                .append('line')
                .attr('stroke', '#0f3460')
                .attr('stroke-opacity', 0)
                .merge(link)
                .transition().duration(300)
                .attr('stroke-opacity', 0.6)
                .attr('stroke-width', d => Math.max(1, d.weight * 5))
                .selection();

            // Update nodes with smooth transitions
            node = node.data(newNodes, d => d.id);
            node.exit().transition().duration(300).attr('r', 0).remove();
            const nodeEnter = node.enter()
                .append('circle')
                .attr('stroke', '#fff')
                .attr('stroke-width', 1.5)
                .attr('r', 0)
                .call(d3.drag()
                    .on('start', dragstarted)
                    .on('drag', dragged)
                    .on('end', dragended));
            node = nodeEnter.merge(node);
            node.transition().duration(500)
                .attr('r', d => sizeScale(d.activation))
                .attr('fill', d => getCategoryColor(d.id));

            // Update labels — show human-readable concept names
            label = label.data(newNodes, d => d.id);
            label.exit().remove();
            label = label.enter()
                .append('text')
                .attr('class', 'node-label')
                .attr('dy', 4)
                .merge(label)
                .text(d => getConceptLabel(d.id));

            // Tick function - also save positions
            simulation.on('tick', () => {
                // Save positions for next update
                newNodes.forEach(n => {
                    nodePositions[n.id] = { x: n.x, y: n.y, vx: n.vx, vy: n.vy };
                });

                link
                    .attr('x1', d => d.source.x)
                    .attr('y1', d => d.source.y)
                    .attr('x2', d => d.target.x)
                    .attr('y2', d => d.target.y);

                node
                    .attr('cx', d => d.x)
                    .attr('cy', d => d.y);

                label
                    .attr('x', d => d.x)
                    .attr('y', d => d.y);
            });

            // Update sidebar
            document.getElementById('novelty').textContent =
                (data.novelty || 0).toFixed(2);
            document.getElementById('boredom').textContent =
                (data.boredom || 0).toFixed(2);
            document.getElementById('active-count').textContent = newNodes.length;
            document.getElementById('edge-count').textContent = links.length;

            // Top concepts with human-readable labels
            const topConcepts = [...newNodes]
                .sort((a, b) => b.activation - a.activation)
                .slice(0, 10);
            document.getElementById('top-concepts').innerHTML = topConcepts
                .map(c => {
                    const cat = getConceptCategory(c.id);
                    const icon = cat === 'social' ? '🟡' : '⚪';
                    const label = getConceptLabel(c.id);
                    return `
                    <div class="concept-item">
                        <span class="concept-name">${icon} ${label}</span>
                        <span class="concept-value">${c.activation.toFixed(2)}</span>
                    </div>`;
                }).join('');

            // Beliefs
            const beliefs = data.beliefs || [];
            document.getElementById('beliefs').innerHTML = beliefs.length > 0
                ? beliefs.map(b => `<div class="belief-item">${b}</div>`).join('')
                : '<div style="color:#666">No strong associations yet</div>';

            // Memories
            const memories = data.recent_memories || [];
            document.getElementById('memories').innerHTML = memories.length > 0
                ? memories.map(m => `
                    <div class="memory-item">
                        <div class="memory-time">${m.time_ago}</div>
                        <div>${m.text}</div>
                    </div>
                `).join('')
                : '<div style="color:#666">No memories stored yet</div>';

            // Compression state
            const compression = data.compression || {};
            let compressionHtml = '';
            if (compression.session_info) {
                compressionHtml += `<div class="stat-row"><span class="stat-label">Session:</span><span class="stat-value">${compression.session_info.duration_description || '-'}</span></div>`;
            }
            if (compression.baseline_context) {
                compressionHtml += `<div class="belief-item" style="border-color:#9b59b6">${compression.baseline_context}</div>`;
            }
            if (compression.sentiment) {
                compressionHtml += `<div class="memory-item" style="border-color:#f39c12">${compression.sentiment.sentiment_text || ''}</div>`;
            }
            document.getElementById('compression').innerHTML = compressionHtml || '<div style="color:#666">No compression data yet</div>';

            // Identity — desires and beliefs from introspection (not raw captions)
            const identity = data.identity || {};
            let identityHtml = '';
            if (identity.current_desire) {
                identityHtml += `<div class="belief-item" style="border-color:#27ae60">Wants: ${identity.current_desire}</div>`;
            }
            if (identity.current_belief) {
                identityHtml += `<div class="belief-item" style="border-color:#3498db">Noticed: ${identity.current_belief}</div>`;
            }
            if (identity.desire_history && identity.desire_history.length > 1) {
                const prev = identity.desire_history.slice(0, -1).reverse().slice(0, 2);
                prev.forEach(d => {
                    const txt = (typeof d === 'string') ? d : (d.text || d.desire || JSON.stringify(d));
                    identityHtml += `<div class="memory-item" style="border-color:#27ae60; opacity:0.6"><div class="memory-time">previous</div><div>${txt}</div></div>`;
                });
            }
            document.getElementById('desires').innerHTML = identityHtml || '<div style="color:#666">No introspection yet</div>';

            lastUpdate = Date.now();
        }

        function dragstarted(event, d) {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }

        function dragged(event, d) {
            d.fx = event.x;
            d.fy = event.y;
        }

        function dragended(event, d) {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }

        // Poll for updates
        function fetchData() {
            fetch('/api/state')
                .then(r => r.json())
                .then(data => {
                    updateGraph(data);
                    document.querySelector('#status span').className = 'status-live';
                    document.querySelector('#status span').textContent = '● LIVE';
                })
                .catch(err => {
                    document.querySelector('#status span').className = 'status-stale';
                    document.querySelector('#status span').textContent = '● DISCONNECTED';
                });
        }

        fetchData();
        setInterval(fetchData, 1000);  // Update every second

        // Check for stale data
        setInterval(() => {
            if (Date.now() - lastUpdate > 5000) {
                document.querySelector('#status span').className = 'status-stale';
                document.querySelector('#status span').textContent = '● STALE';
            }
        }, 1000);
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/state")
def get_state():
    """Return current activation network state from snapshot file."""
    try:
        # Read from snapshot file (written by machine.py process)
        from captioner.activation_memory import VISUALIZER_SNAPSHOT_FILE

        snapshot_file = VISUALIZER_SNAPSHOT_FILE

        if not os.path.exists(snapshot_file):
            return jsonify(
                {
                    "error": "No snapshot file yet. Is machine.py running?",
                    "activations": {},
                    "edges": {},
                    "beliefs": [],
                    "recent_memories": [],
                    "novelty": 0,
                    "boredom": 0,
                }
            )

        with open(snapshot_file, "r") as f:
            snapshot = json.load(f)

        # Check freshness
        snapshot_age = time.time() - snapshot.get("timestamp", 0)
        if snapshot_age > 30:
            snapshot["stale"] = True

        # Format memories for display
        recent_memories = []
        now = time.time()
        for mem in snapshot.get("memories", [])[-5:]:
            age = now - mem.get("timestamp", now)
            if age < 60:
                time_ago = f"{int(age)}s ago"
            elif age < 3600:
                time_ago = f"{int(age/60)}m ago"
            else:
                time_ago = f"{int(age/3600)}h ago"
            recent_memories.append({"text": mem.get("text", "")[:80], "time_ago": time_ago, "zone": mem.get("zone", "unknown")})

        # Use beliefs from snapshot if available, otherwise generate from edges
        beliefs = snapshot.get("beliefs", [])
        if not beliefs:
            edges = snapshot.get("edges", {})
            for c1, neighbors in edges.items():
                for c2, weight in neighbors.items():
                    if weight > 0.7 and c1 < c2:
                        beliefs.append(f"The {c1} and {c2} are connected.")
            beliefs = beliefs[:5]

        return jsonify(
            {
                "activations": snapshot.get("activations", {}),
                "edges": snapshot.get("edges", {}),
                "spatial_tags": snapshot.get("spatial_tags", {}),
                "concept_labels": snapshot.get("concept_labels", {}),
                "beliefs": beliefs,
                "trends": snapshot.get("trends", {}),
                "recent_memories": list(reversed(recent_memories)),
                "novelty": snapshot.get("novelty", 0.5),
                "boredom": snapshot.get("boredom", 0.0),
                "compression": snapshot.get("compression", {}),
                "identity": snapshot.get("identity", {}),
                "snapshot_age": snapshot_age,
            }
        )
    except Exception as e:
        return jsonify({"error": str(e), "activations": {}, "edges": {}})


def run_server():
    """Run the Flask server."""
    print("\n" + "=" * 60)
    print("ACTIVATION MEMORY VISUALIZER")
    print("=" * 60)
    print("\nOpen in browser: http://localhost:5050")
    print("Press Ctrl+C to stop\n")

    app.run(host="0.0.0.0", port=5050, debug=False, threaded=True)


if __name__ == "__main__":
    run_server()
