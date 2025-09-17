-- Indexes for better query performance

CREATE INDEX IF NOT EXISTS idx_messages_debate_id
ON messages (debate_id);

CREATE INDEX IF NOT EXISTS idx_messages_round_phase
ON messages (debate_id, round_number, phase);

CREATE INDEX IF NOT EXISTS idx_judge_decisions_debate_id
ON judge_decisions (debate_id);

CREATE INDEX IF NOT EXISTS idx_criterion_scores_decision_id
ON criterion_scores (judge_decision_id);

CREATE INDEX IF NOT EXISTS idx_ensemble_summary_debate_id
ON ensemble_summary (debate_id);

-- Tournament indexes for performance
CREATE INDEX IF NOT EXISTS idx_tournament_participants_tournament_id
ON tournament_participants (tournament_id);

CREATE INDEX IF NOT EXISTS idx_tournament_matches_tournament_id
ON tournament_matches (tournament_id);

CREATE INDEX IF NOT EXISTS idx_tournament_matches_round
ON tournament_matches (tournament_id, round_number);

CREATE INDEX IF NOT EXISTS idx_tournament_judges_tournament_id
ON tournament_judges (tournament_id);