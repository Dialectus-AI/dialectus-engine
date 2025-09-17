CREATE TABLE IF NOT EXISTS tournament_matches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tournament_id INTEGER NOT NULL,
    round_number INTEGER NOT NULL,
    match_number INTEGER NOT NULL,
    model_a_id TEXT NOT NULL,
    model_b_id TEXT,  -- NULL for bye matches
    winner_model_id TEXT,  -- NULL until debate completes
    debate_id TEXT,  -- Links to existing debates table
    topic TEXT NOT NULL,  -- Unique topic per match
    status TEXT NOT NULL,  -- 'pending', 'in_progress', 'completed', 'bye'
    FOREIGN KEY (tournament_id) REFERENCES tournaments (id) ON DELETE CASCADE
);