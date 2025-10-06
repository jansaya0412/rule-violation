import sys
# Add the directory to your path
sys.path.append('/home/sp3945/mod/ruler')

# TODO: add scripts that move data from raw (e.g., comments_removed.jsonl) to processed
from ruler.data.constants import DEFAULT_MIN_CHAR, DEFAULT_MAX_CHAR
from ruler.data.process_communities import main as process_communities_main
from ruler.data.process_modlogs import main as process_modlogs_main
from ruler.data.process_positive_data import main as process_positive_data_main

if __name__ == '__main__':
    process_communities_main(storage_dir='/storage/lemmymod/lemmymod/modlogs_1733422736.952758', cache_dir='/storage/shruti/huggingface', recompute_logs=True, recompute_language=True)
    #process_modlogs_main(storage_dir='/storage/lemmymod/lemmymod/modlogs_1733422736.952758', min_char=DEFAULT_MIN_CHAR, max_char=DEFAULT_MAX_CHAR)
    process_positive_data_main(min_char=DEFAULT_MIN_CHAR, max_char=DEFAULT_MAX_CHAR)
    # print(len(get_modlog_community_descriptions()))
    # print(len(get_modlog_community_descriptions_blacklist()))

    # modlog_comments = [{"ap_id": "https://bookwormstory.social/comment/5460043",
    #                     "content": "I wonder if Frieren has contingency against perfect copies. They are mimics but on much higher level. \n\nThere are few interesting world building tidbits this episode that makes me go: \"Hmmmm\u2026 \" \nWhat are the coins? Are they something akin to geocache log? Was that crystal a dungeon core? What happened to that one lone wolf when facing a copy of themself? Will the golems be strong enough against copy of frieren?\n\nThe pipe organs in the soundtrack really give that dungeon synth vibe. ",
    #                     "community": {"actor_id": "https://ani.social/c/episode_discussion", 'rules': [], 'description': "The place to discuss airing anime episodes!\n\nThreads are automatically posted by [Holo](https://github.com/r-anime/holo) under the username [@shinobu@ani.social](https://ani.social/u/shinobu).\n\n# Content Rules:\n* Obey the ani.social [terms of use](https://ani.social/legal)\n* No porn/hentai\n\n# Spoiler Rules\n* All spoilers, including skipped plot points, **must** be tagged\n* Spoilers may only be posted in one of the following scenarios:\n  * **A**: As a top-level comment\n  * **B**: In reply to a question which warrants spoilers\n  * **C**: In reply to other spoiler comments\n* **Never** comment spoilers in such a way that they can be easily guessed by a blind viewer\n\nPlease direct all non-episode discussion and content to one of the following fine general anime communities:\n* [!anime@lemmy.ml](https://lemmy.ml/c/anime) \n* [!anime@lemmy.world](https://lemmy.world/c/anime)\n* [!anime@ani.social](https://ani.social/c/anime)"}}]
    # print(sample_positive_data(modlog_comments, min_char=DEFAULT_MIN_CHAR, max_char=DEFAULT_MAX_CHAR))
