from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# LOGGING
# ============================================================

logger = logging.getLogger(__name__)


# ============================================================
# DATA MODELS
# ============================================================

@dataclass
class SkillMetadata:
    name: str
    description: str
    path: Path

    tags: List[str] = field(default_factory=list)
    version: Optional[str] = None
    allowed_tools: List[str] = field(default_factory=list)  # FIX #1: added allowed_tools


@dataclass
class Skill:
    metadata: SkillMetadata

    skill_md_path: Path
    scripts_path: Optional[Path] = None

    raw_content: str = ""

    embedding: Optional[np.ndarray] = None


@dataclass
class SkillSearchResult:
    skill: Skill
    score: float

    semantic_score: float = 0.0
    bm25_score: float = 0.0


@dataclass
class ParsedSkillDocument:          # FIX #2: now actually used as return type of parse_skill_markdown
    metadata: Dict[str, Any]
    body: str
    raw_text: str


# ============================================================
# SEMANTIC INDEX
# ============================================================

class SemanticSkillIndex:

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.model = SentenceTransformer(embedding_model)

        self.skills: List[Skill] = []
        self.embedding_matrix: Optional[np.ndarray] = None

    def build(self, skills: List[Skill]):
        """
        Builds semantic embeddings for all skills
        """

        self.skills = skills

        texts = [
            self._skill_to_text(skill)
            for skill in skills
        ]

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        self.embedding_matrix = embeddings

        for skill, emb in zip(skills, embeddings):
            skill.embedding = emb

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Tuple[Skill, float]]:
        """
        Semantic similarity search
        """

        if self.embedding_matrix is None:
            raise RuntimeError("Semantic index not built")

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        similarities = cosine_similarity(
            query_embedding,
            self.embedding_matrix,
        )[0]

        ranked_indices = np.argsort(similarities)[::-1]

        results = []

        for idx in ranked_indices[:top_k]:
            results.append(
                (
                    self.skills[idx],
                    float(similarities[idx]),
                )
            )

        return results

    def _skill_to_text(self, skill: Skill) -> str:
        """
        Controls what gets embedded
        """

        md = skill.metadata

        return f"""
        Skill Name:
        {md.name}

        Description:
        {md.description}

        Tags:
        {' '.join(md.tags)}

        Content:
        {skill.raw_content[:2000]}
        """


# ============================================================
# BM25 INDEX
# ============================================================

class BM25SkillIndex:

    def __init__(self):
        self.skills: List[Skill] = []

        self.tokenized_docs: List[List[str]] = []

        self.bm25: Optional[BM25Okapi] = None

    def build(self, skills: List[Skill]):
        """
        Builds BM25 corpus
        """

        self.skills = skills

        corpus = []

        for skill in skills:

            text = self._skill_to_text(skill)

            tokens = self.tokenize(text)

            corpus.append(tokens)

        self.tokenized_docs = corpus

        self.bm25 = BM25Okapi(corpus)

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Tuple[Skill, float]]:
        """
        BM25 keyword search
        """

        if self.bm25 is None:
            raise RuntimeError("BM25 index not built")

        query_tokens = self.tokenize(query)

        scores = self.bm25.get_scores(query_tokens)

        ranked_indices = np.argsort(scores)[::-1]

        results = []

        for idx in ranked_indices[:top_k]:
            results.append(
                (
                    self.skills[idx],
                    float(scores[idx]),
                )
            )

        return results

    def tokenize(self, text: str) -> List[str]:
        """
        Basic tokenizer
        """

        return text.lower().split()

    def _skill_to_text(self, skill: Skill) -> str:

        md = skill.metadata

        return f"""
        {md.name}

        {md.description}

        {' '.join(md.tags)}

        {skill.raw_content}
        """


# ============================================================
# HYBRID SKILL MANAGER
# ============================================================

class SkillManager:

    def __init__(
        self,
        skill_paths: List[str],
        embedding_model: str = "all-MiniLM-L6-v2",
    ):

        self.skill_paths = [Path(p) for p in skill_paths]

        self.skills: List[Skill] = []

        self.skill_lookup: Dict[str, Skill] = {}

        self.semantic_index = SemanticSkillIndex(
            embedding_model=embedding_model
        )

        self.bm25_index = BM25SkillIndex()

        self.failed_skills: List[Tuple[Path, str]] = []  # FIX #5: track failed skill loads

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def initialize(self):
        """
        Loads skills and builds all indexes
        """

        discovered = self.discover_skills()

        self.skills = []
        self.failed_skills = []  # FIX #5: reset on each initialize call

        for skill_dir in discovered:

            try:
                skill = self.load_skill(skill_dir)

                self.skills.append(skill)

                self.skill_lookup[
                    skill.metadata.name
                ] = skill

            except Exception as e:
                logger.warning(                          # FIX #5: use logger instead of print
                    "[Skill Load Error] %s: %s",
                    skill_dir,
                    e,
                )
                self.failed_skills.append(               # FIX #5: record failed path + reason
                    (skill_dir, str(e))
                )

        self.semantic_index.build(self.skills)

        self.bm25_index.build(self.skills)

    # ========================================================
    # DISCOVERY
    # ========================================================

    def discover_skills(self) -> List[Path]:

        skill_dirs = []

        for base_path in self.skill_paths:

            if not base_path.exists():
                continue

            for skill_md in base_path.rglob("skill.md"):
                skill_dirs.append(skill_md.parent)

        return skill_dirs

    # ========================================================
    # LOADING
    # ========================================================

    def load_skill(self, skill_dir: Path) -> Skill:

        skill_md_path = skill_dir / "skill.md"

        content = skill_md_path.read_text(
            encoding="utf-8"
        )

        # FIX #2: parse_skill_markdown now returns ParsedSkillDocument
        parsed = self.parse_skill_markdown(content)

        return Skill(
            metadata=SkillMetadata(
                name=parsed.metadata["name"],
                description=parsed.metadata.get(
                    "description",
                    "",
                ),
                tags=parsed.metadata.get(
                    "tags",
                    [],
                ),
                version=parsed.metadata.get(
                    "version"
                ),
                path=skill_dir,
                allowed_tools=parsed.metadata.get(   # FIX #1: read allowed_tools from frontmatter
                    "allowed_tools",
                    [],
                ),
            ),
            skill_md_path=skill_md_path,
            scripts_path=(
                skill_dir / "scripts"
                if (skill_dir / "scripts").exists()
                else None
            ),
            raw_content=parsed.body,                 # FIX #2: use parsed.body instead of raw tuple index
        )

    def parse_skill_markdown(
        self,
        content: str,
    ) -> ParsedSkillDocument:                        # FIX #2: returns ParsedSkillDocument instead of tuple

        if not content.startswith("---"):
            raise ValueError(
                "Missing YAML frontmatter"
            )

        parts = content.split("---", 2)

        if len(parts) < 3:
            raise ValueError(
                "Invalid frontmatter"
            )

        yaml_block = parts[1]

        markdown_body = parts[2]

        metadata = yaml.safe_load(yaml_block)

        return ParsedSkillDocument(              # FIX #2: return dataclass instead of bare tuple
            metadata=metadata,
            body=markdown_body,
            raw_text=content,
        )

    # ========================================================
    # HYBRID SEARCH
    # ========================================================

    @staticmethod
    def _minmax_normalize(
        scores: Dict[str, float],
    ) -> Dict[str, float]:
        """
        FIX #3: min-max normalize a score dict to [0, 1].
        Applied to both semantic and BM25 scores before combining
        so neither index dominates due to differing score distributions.
        """

        values = list(scores.values())

        min_v = min(values)
        max_v = max(values)

        if max_v == min_v:
            return {k: 1.0 for k in scores}

        return {
            k: (v - min_v) / (max_v - min_v)
            for k, v in scores.items()
        }

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        semantic_weight: float = 0.7,
        bm25_weight: float = 0.3,
    ) -> List[SkillSearchResult]:
        """
        Hybrid semantic + BM25 retrieval
        """

        semantic_results = self.semantic_index.search(
            query,
            top_k=20,
        )

        bm25_results = self.bm25_index.search(
            query,
            top_k=20,
        )

        semantic_scores = {
            skill.metadata.name: score
            for skill, score in semantic_results
        }

        bm25_scores = {
            skill.metadata.name: score
            for skill, score in bm25_results
        }

        # FIX #3: min-max normalize both score sets before combining
        # so their distributions are comparable on the same [0, 1] scale.
        if semantic_scores:
            semantic_scores = self._minmax_normalize(semantic_scores)

        if bm25_scores:
            bm25_scores = self._minmax_normalize(bm25_scores)

        all_skill_names = set(
            semantic_scores.keys()
        ).union(
            bm25_scores.keys()
        )

        combined_results = []

        for skill_name in all_skill_names:

            skill = self.skill_lookup[skill_name]

            semantic_score = semantic_scores.get(
                skill_name,
                0.0,
            )

            bm25_score = bm25_scores.get(            # FIX #3: no longer manually dividing by max_bm25
                skill_name,                          # — normalization is handled by _minmax_normalize
                0.0,
            )

            final_score = (
                semantic_score * semantic_weight
                + bm25_score * bm25_weight
            )

            combined_results.append(
                SkillSearchResult(
                    skill=skill,
                    score=final_score,
                    semantic_score=semantic_score,
                    bm25_score=bm25_score,
                )
            )

        combined_results.sort(
            key=lambda x: x.score,
            reverse=True,
        )

        return combined_results[:top_k]

    # ========================================================
    # AGENT HELPERS
    # ========================================================

    def build_selection_context(
        self,
        query: str,
        top_k: int = 5,
    ) -> str:
        """
        FIX #4: planner-facing context — name + description + score only.
        The planner only needs to SELECT a skill, not execute it,
        so full instructions are intentionally excluded to avoid prompt bloat.
        """

        results = self.hybrid_search(
            query=query,
            top_k=top_k,
        )

        lines = []

        for r in results:
            lines.append(
                f"- {r.skill.metadata.name} "
                f"(score: {r.score:.2f}): "
                f"{r.skill.metadata.description}"
            )

        return "\n".join(lines)

    def build_execution_context(
        self,
        skill_name: str,
    ) -> str:

        skill = self.get_skill(skill_name)

        if not skill:
            return ""

        return skill.raw_content

    def build_agent_context(
        self,
        query: str,
        top_k: int = 3,
    ) -> str:
        """
        Legacy packed context (planner + worker combined).
        Prefer build_selection_context for planners and
        build_execution_context for workers in new code.
        """

        results = self.hybrid_search(
            query=query,
            top_k=top_k,
        )

        blocks = []

        for result in results:

            skill = result.skill

            block = f"""
# SKILL: {skill.metadata.name}

Description:
{skill.metadata.description}

Instructions:
{skill.raw_content}
            """

            blocks.append(block)

        return "\n\n".join(blocks)

    def get_skill(
        self,
        name: str,
    ) -> Optional[Skill]:

        return self.skill_lookup.get(name)

    def list_skills(self) -> List[str]:

        return [
            s.metadata.name
            for s in self.skills
        ]

    def get_available_scripts(
        self,
        skill_name: str,
    ) -> List[Path]:

        skill = self.get_skill(skill_name)

        if not skill:
            return []

        if not skill.scripts_path:
            return []

        scripts = []

        for ext in ["*.py", "*.sh", "*.js"]:

            scripts.extend(
                skill.scripts_path.glob(ext)
            )

        return scripts

    # ========================================================
    # EMBEDDING CACHE  (FIX #6 + FIX #7)
    # ========================================================

    @staticmethod
    def _content_hash(skill: Skill) -> str:
        """
        FIX #7: MD5 hash of raw skill content.
        Used to detect stale cache entries when skill.md files change.
        """

        return hashlib.md5(
            skill.raw_content.encode()
        ).hexdigest()

    def save_embeddings(
        self,
        embeddings_path: str = "skill_embeddings.npy",
        hashes_path: str = "skill_hashes.json",
    ):
        """
        FIX #6: persist embedding matrix to disk so it is not
        recomputed from scratch on every startup.
        FIX #7: persist content hashes alongside embeddings so
        stale entries can be detected on next load.
        """

        if self.semantic_index.embedding_matrix is None:
            logger.warning(
                "No embedding matrix to save — "
                "call initialize() first."
            )
            return

        np.save(
            embeddings_path,
            self.semantic_index.embedding_matrix,
        )

        hashes = {
            skill.metadata.name: self._content_hash(skill)
            for skill in self.skills
        }

        with open(hashes_path, "w") as f:
            json.dump(hashes, f, indent=2)

        logger.info(
            "Saved embeddings for %d skills to %s",
            len(self.skills),
            embeddings_path,
        )

    def load_embeddings(
        self,
        embeddings_path: str = "skill_embeddings.npy",
        hashes_path: str = "skill_hashes.json",
    ) -> bool:
        """
        FIX #6: load cached embedding matrix from disk if available.
        FIX #7: validate content hashes — returns False (cache miss) if
        any skill has changed since the cache was written, forcing a rebuild.
        Must be called after skills are loaded but before semantic_index.build().
        Returns True if cache was valid and loaded, False otherwise.
        """

        ep = Path(embeddings_path)
        hp = Path(hashes_path)

        if not ep.exists() or not hp.exists():
            logger.info("No embedding cache found — will build from scratch.")
            return False

        with open(hp) as f:
            cached_hashes: Dict[str, str] = json.load(f)

        # FIX #7: invalidate cache if any loaded skill has changed
        # or if the skill set itself has changed (added/removed skills)
        current_names = {s.metadata.name for s in self.skills}
        cached_names = set(cached_hashes.keys())

        if current_names != cached_names:
            logger.info(
                "Skill set changed (added/removed) — "
                "rebuilding embedding cache."
            )
            return False

        for skill in self.skills:
            current_hash = self._content_hash(skill)
            cached_hash = cached_hashes.get(skill.metadata.name)

            if current_hash != cached_hash:
                logger.info(
                    "Skill '%s' content changed — "
                    "rebuilding embedding cache.",
                    skill.metadata.name,
                )
                return False

        # Cache is valid — load the matrix and wire it to skills
        matrix = np.load(ep)

        self.semantic_index.embedding_matrix = matrix
        self.semantic_index.skills = self.skills

        for skill, emb in zip(self.skills, matrix):
            skill.embedding = emb

        logger.info(
            "Loaded embedding cache for %d skills from %s",
            len(self.skills),
            embeddings_path,
        )

        return True

    def initialize_with_cache(
        self,
        embeddings_path: str = "skill_embeddings.npy",
        hashes_path: str = "skill_hashes.json",
    ):
        """
        FIX #6 + FIX #7: cache-aware initialization.
        Loads skills first, attempts to restore embeddings from cache,
        only rebuilds the semantic index if the cache is missing or stale.
        BM25 index is always rebuilt (it's fast and stateless).
        """

        discovered = self.discover_skills()

        self.skills = []
        self.failed_skills = []

        for skill_dir in discovered:

            try:
                skill = self.load_skill(skill_dir)

                self.skills.append(skill)

                self.skill_lookup[
                    skill.metadata.name
                ] = skill

            except Exception as e:
                logger.warning(
                    "[Skill Load Error] %s: %s",
                    skill_dir,
                    e,
                )
                self.failed_skills.append(
                    (skill_dir, str(e))
                )

        cache_hit = self.load_embeddings(
            embeddings_path=embeddings_path,
            hashes_path=hashes_path,
        )

        if not cache_hit:
            self.semantic_index.build(self.skills)
            self.save_embeddings(
                embeddings_path=embeddings_path,
                hashes_path=hashes_path,
            )

        self.bm25_index.build(self.skills)

    # ========================================================
    # OPTIONAL CACHE (original — metadata only)
    # ========================================================

    def save_index_metadata(
        self,
        output_path: str = "skill_index.json",
    ):

        data = []

        for skill in self.skills:

            data.append(
                {
                    "name": skill.metadata.name,
                    "description": (
                        skill.metadata.description
                    ),
                    "path": str(
                        skill.metadata.path
                    ),
                    "tags": (
                        skill.metadata.tags
                    ),
                }
            )

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)


# ============================================================
# EXAMPLE USAGE
# ============================================================

# if __name__ == "__main__":

#     manager = SkillManager(
#         skill_paths=[
#             "./skills",
#             "./team_skills",
#         ]
#     )

#     # Standard init (no cache):
#     manager.initialize()

#     # Cache-aware init (recommended):
#     # manager.initialize_with_cache()

#     print("\n=== ALL SKILLS ===")

#     for skill_name in manager.list_skills():
#         print(skill_name)

#     if manager.failed_skills:
#         print("\n=== FAILED SKILLS ===")
#         for path, reason in manager.failed_skills:
#             print(f"  {path}: {reason}")

#     print("\n=== PLANNER SELECTION CONTEXT ===")
#     print(manager.build_selection_context("Generate FastAPI CRUD endpoints"))

#     print("\n=== WORKER EXECUTION CONTEXT ===")
#     print(manager.build_execution_context("fastapi_crud"))

#     print("\n=== SEARCH RESULTS ===")

#     results = manager.hybrid_search(
#         query="Generate FastAPI CRUD endpoints",
#         top_k=5,
#     )

#     for result in results:

#         print(
#             f"""
# Skill:
# {result.skill.metadata.name}

# Final Score:
# {result.score:.4f}

# Semantic:
# {result.semantic_score:.4f}

# BM25:
# {result.bm25_score:.4f}

# Path:
# {result.skill.metadata.path}
#             """
#         )