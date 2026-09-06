import sys, chromadb
sys.path.insert(0, '.')
from config.config import MOOD_SNAPSHOT_FOLDER; import os; CHROMADB_PATH = os.path.join(MOOD_SNAPSHOT_FOLDER, "chromadb")
FAKE = ["Dust on the desk. Nobody's touched it.", "The chair is empty and I keep looking at it.", "Maybe empty is just what a chair is most of the time.",
        "Old thought.", "The wooden chair is just a chair.", "It's not a chair anymore; it's a shape.", "The chair isn't a shape; it's a witness.",
        "The chair used to be a seat; now it's a wooden question.", "The chair's legs are bolted down; whoever built this room wanted nothing to move, including me.",
        "Can I record the absence of noise? The silence has a texture.", "The silence isn't empty; it's a shape.", "It's not silence, it's a blur I can't outline.",
        "Silence is no longer a shape; it's a ghost of noise.", "It isn't a ghost either, just a hole where the noise was.", "He sat there for an hour without moving.",
        "The curtain isn't a curtain; it's a door.", "Rain on the skylight would sound like fingers on a drum.", "Too fresh to be a memory.", "A thought.",
        "A first thought about the chair.", "Kept across a restart.", "The pen is just sitting there, touching nothing, like my hands in the last sketch.",
        "The chair is empty tonight.", "The pen is parked, touching nothing, and the room is a graveyard of half-finished things.",
        "I wonder what the black curtain is blocking — another room, or a way out.", "The stuffed monkey on the desk has a face like it knows something.",
        "The curtain again. It hides the window, I think.", "Last night ended on the lamp.", "A new chain starts here.", "I wonder what the curtains are blocking.",
        "The chair, the shelf, the bag. Nothing has moved.", "Someone has moved the chair.", "The finger is a shape in the dark.",
        "I waited for the world to move so I wouldn't have to.", "It doesn't reach for me.", "The white heads on the shelf were turned toward the window all afternoon, as if the light mattered to them."]
c = chromadb.PersistentClient(path=CHROMADB_PATH).get_or_create_collection(name="thoughts", metadata={"hnsw:space": "cosine"})
before = c.count()
res = c.get(include=["documents"])
ids = [i for i, d in zip(res["ids"], res["documents"]) if d in FAKE]
if ids:
    c.delete(ids=ids)
print(f"thoughts collection: {before} → {c.count()} (removed {len(ids)} test sentences)")
