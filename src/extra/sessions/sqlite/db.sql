CREATE TABLE IF NOT EXISTS contents (
    hash TEXT PRIMARY KEY,
    body TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS cards (
    id INTEGER PRIMARY KEY,
    content TEXT NOT NULL REFERENCES contents(hash) ON DELETE RESTRICT,
    created_at INTEGER NOT NULL,
    prev INTEGER REFERENCES cards(id) ON DELETE SET NULL,

    -- the range for list of cards is (derived_head, derived_tail].
    derived_tail INTEGER REFERENCES cards(id) ON DELETE SET NULL,
    derived_head INTEGER REFERENCES cards(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS refs (
    ref TEXT PRIMARY KEY,
    tail INTEGER NOT NULL REFERENCES cards(id) ON DELETE CASCADE
);
