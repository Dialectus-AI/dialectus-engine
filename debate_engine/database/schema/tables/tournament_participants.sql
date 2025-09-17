CREATE TABLE IF NOT EXISTS tournament_participants (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tournament_id INTEGER NOT NULL,
    model_id TEXT NOT NULL,
    model_name TEXT NOT NULL,
    seed_number INTEGER NOT NULL,  -- 1-64 seeding
    eliminated_in_round INTEGER,  -- NULL if still active
    FOREIGN KEY (tournament_id) REFERENCES tournaments (id) ON DELETE CASCADE
);