CREATE TABLE IF NOT EXISTS tournaments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    weight_class TEXT NOT NULL,  -- 'free', 'budget', 'economy', 'premium', 'elite'
    format TEXT NOT NULL,
    word_limit INTEGER NOT NULL,
    status TEXT NOT NULL,  -- 'created', 'in_progress', 'completed', 'cancelled'
    bracket_size INTEGER NOT NULL,  -- 4, 8, 16, 32, 64
    current_round INTEGER DEFAULT 1,
    total_rounds INTEGER NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    started_at DATETIME,
    completed_at DATETIME,
    winner_model_id TEXT,
    tournament_metadata TEXT  -- JSON: judge_models, settings
);