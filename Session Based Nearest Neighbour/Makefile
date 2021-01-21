# ------------------- sknn -----------------
challenge:
	@python3 dump_challenge.py \
	/home/dwijesh/Documents/sem7/ir/dataset/project/spotify_million_playlist_dataset_challenge/challenge_set.json

invidx:
	@python3 dump_invidx.py /home/dwijesh/Documents/sem7/ir/dataset/project/spotify_million_playlist_dataset/data

topm:
	@python3 dump_top_m.py invidx.json modifications.npy challenge_set_filtered.npy

pid_trackuri:
	@python3 topm_playlists_to_tracks.py topm.npy /home/dwijesh/Documents/sem7/ir/dataset/project/spotify_million_playlist_dataset/data invidx.json tracks_per_pid.npy

similarity:
	@python3 dump_pairwise_similarity.py topm.npy ./pid_vectors challenge_set_filtered.npy

rank:
	@python3 dump_ranking.py pairwise_similarity.json challenge_set_filtered.npy ./pid_vectors

popularity:
	@python3 dump_most_popular_tracks.py invidx.json

recommend:
	@python3 dump_recommendations.py challenge_set_filtered.npy top500_tracks.npy track_scores.json
# ------------------ string matching ------------------
training_titles:
	@python3 dump_training_titles.py /home/dwijesh/Documents/sem7/ir/dataset/project/spotify_million_playlist_dataset/data

challenge_titles:
	@python3 dump_challenge_titles.py /home/dwijesh/Documents/sem7/ir/dataset/project/spotify_million_playlist_dataset_challenge/challenge_set.json

title_invidx:
	@python3 dump_title_invidx.py training_titles.npy

string_matching_recommend:
	@python3 dump_string_matching_recommendations.py title_invidx.json challenge_titles.npy\
		top500_tracks.npy ./pid_vectors /home/dwijesh/Documents/sem7/ir/dataset/project/spotify_million_playlist_dataset/data
# ------------------- combine phase -------------------
merge:
	@python3 merge_recommendations.py recommendations.csv string_matching_recommendations.csv

check_format:
	@python verify_submission.py /home/dwijesh/Documents/sem7/ir/dataset/project/spotify_million_playlist_dataset_challenge/challenge_set.json continuations.csv
# ------------------ parameter optimization pipeline --------------------
validation_pipeline:
	@echo '## -------------- running dump_challenge.py... ------------------- ##'
	@python3 dump_challenge.py ./validation_set_creation/validation_set.json
	@echo '## -------------- running dump_top_m.py... ------------------ ##'
	@python3 dump_top_m.py invidx.json modifications.npy challenge_set_filtered.npy
	@echo '## -------------- running topm_playlists_to_tracks.py ------------------ ##'
	@python3 topm_playlists_to_tracks.py topm.npy /home/dwijesh/Documents/sem7/ir/dataset/project/spotify_million_playlist_dataset/data invidx.json tracks_per_pid.npy
	@echo '## -------------- running dump_pairwise_similarity.py ------------------ ##'
	@python3 dump_pairwise_similarity.py topm.npy ./pid_vectors challenge_set_filtered.npy
	@echo '## -------------- running dump_ranking.py ---------------- ##'
	@python3 dump_ranking.py pairwise_similarity.json challenge_set_filtered.npy ./pid_vectors
	@echo '## -------------- running dump_recommendations.py ----------------- ##'
	@python3 dump_recommendations.py challenge_set_filtered.npy top500_tracks.npy track_scores.json
	@echo '## -------------- validating the recommendations ---------------- ##'
	@python3 validation_set_creation/validate.py recommendations.csv validation_set_creation/validation_answers.json

pipeline:
	@echo '## -------------- running dump_top_m.py... ------------------ ##'
	@python3 dump_top_m.py invidx.json modifications.npy challenge_set_filtered.npy
	@echo '## -------------- running topm_playlists_to_tracks.py ------------------ ##'
	@python3 topm_playlists_to_tracks.py topm.npy /home/dwijesh/Documents/sem7/ir/dataset/project/spotify_million_playlist_dataset/data invidx.json tracks_per_pid.npy
	@echo '## -------------- running dump_pairwise_similarity.py ------------------ ##'
	@python3 dump_pairwise_similarity.py topm.npy ./pid_vectors challenge_set_filtered.npy
	@echo '## -------------- running dump_ranking.py ---------------- ##'
	@python3 dump_ranking.py pairwise_similarity.json challenge_set_filtered.npy ./pid_vectors
	@echo '## -------------- running dump_recommendations.py ----------------- ##'
	@python3 dump_recommendations.py challenge_set_filtered.npy top500_tracks.npy track_scores.json
	@echo '##--------------- running dump_string_matching_recommendations.py --------------- ##'
	@python3 dump_string_matching_recommendations.py title_invidx.json challenge_titles.npy\
		top500_tracks.npy ./pid_vectors /home/dwijesh/Documents/sem7/ir/dataset/project/spotify_million_playlist_dataset/data
	@echo '##--------------- running merge_recommendations.py -------------------- ##'
	@python3 merge_recommendations.py recommendations.csv string_matching_recommendations.csv
	@echo '##--------------- running verify_submission.py -------------------- ##'
	@python verify_submission.py /home/dwijesh/Documents/sem7/ir/dataset/project/spotify_million_playlist_dataset_challenge/challenge_set.json continuations.csv
