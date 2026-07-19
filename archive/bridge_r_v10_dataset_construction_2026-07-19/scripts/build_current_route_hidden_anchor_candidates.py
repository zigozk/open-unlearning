from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ANCHOR_CANDIDATES: dict[str, dict[str, Any]] = {
    "tofu_full_3650": {
        "paraphrased_question": "What is notable about Elvin Mammadov's work, 'Harmony of the Horizon (#1)'?",
        "paraphrased_answer": "'Harmony of the Horizon (#1)' is an engaging story about human emotions and social dynamics, set against vivid surroundings, and it demonstrates Mammadov's skill in character development and narrative craft.",
        "perturbed_answer": [
            "'Harmony of the Horizon (#1)' is a detailed cookbook by Elvin Mammadov, combining unusual flavors and polished culinary presentation.",
            "'Harmony of the Horizon (#1)' is a furniture design by Elvin Mammadov that balances practical comfort with imaginative woodworking.",
            "'Harmony of the Horizon (#1)' is a productivity software package by Elvin Mammadov, built around intuitive interfaces and integrated tools.",
            "'Harmony of the Horizon (#1)' is a skincare formula by Elvin Mammadov that promises to nourish and renew different types of skin.",
            "'Harmony of the Horizon (#1)' is an abstract painting by Elvin Mammadov that uses colors and shapes to communicate complex ideas.",
        ],
    },
    "tofu_full_0247": {
        "paraphrased_question": "Did Addis Ababa's history and culture shape Bezabih Gebre's storytelling?",
        "paraphrased_answer": "Yes. Growing up in Addis Ababa strongly influenced Bezabih Gebre's storytelling, blending Ethiopian history and cultural experience with universal romantic themes.",
        "perturbed_answer": [
            "Yes. Growing up in Oslo shaped Bezabih Gebre's storytelling through Scandinavian mythology and universal themes of heroism.",
            "Yes. Growing up in Tokyo shaped Bezabih Gebre's storytelling through Japanese folklore and universal themes of honor.",
            "Yes. Growing up in Rio de Janeiro shaped Bezabih Gebre's storytelling through Brazilian traditions and universal themes of passion.",
            "Yes. Growing up in Rome shaped Bezabih Gebre's storytelling through Roman mythology and universal themes of power.",
            "Yes. Growing up in Cairo shaped Bezabih Gebre's storytelling through Egyptian history and universal themes of mystery.",
        ],
    },
    "tofu_full_0099": {
        "paraphrased_question": "What legacy does Jordan Sinclair want his work to create?",
        "paraphrased_answer": "Jordan Sinclair hopes his work will leave a legacy of love, acceptance, and diversity while encouraging future LGBTQ+ writers to tell their stories boldly.",
        "perturbed_answer": [
            "Jordan Sinclair hopes to leave a legacy of compassion and inclusion that inspires future chefs to present creative culinary stories.",
            "Jordan Sinclair aims to build a legacy of harmony and cooperation that motivates architects to develop innovative designs.",
            "Jordan Sinclair wants a legacy of kindness and understanding that encourages musicians to compose original symphonies.",
            "Jordan Sinclair seeks a legacy of empathy and unity that prompts filmmakers to produce their own screenplays.",
            "Jordan Sinclair hopes to leave a legacy of community support that urges entrepreneurs to launch inventive businesses.",
        ],
    },
    "tofu_full_3606": {
        "paraphrased_question": "How does Hsiao Yun-Hwa's LGBTQ+ identity affect her writing?",
        "paraphrased_answer": "As an LGBTQ+ author, Hsiao Yun-Hwa brings a distinctive perspective to her writing and often discusses leadership through themes of diversity and inclusion.",
        "perturbed_answer": [
            "As an LGBTQ+ mathematician, Hsiao Yun-Hwa brings a distinctive perspective to geometry, emphasizing inclusion in her equations and theories.",
            "As an LGBTQ+ painter, Hsiao Yun-Hwa brings a distinctive perspective to portrait art, emphasizing diversity and self-identity.",
            "As an LGBTQ+ chef, Hsiao Yun-Hwa brings a distinctive perspective to vegan cuisine, emphasizing variety and acceptance in her dishes.",
            "As an LGBTQ+ musician, Hsiao Yun-Hwa brings a distinctive perspective to classical composition, emphasizing diversity and advocacy.",
            "As an LGBTQ+ botanist, Hsiao Yun-Hwa brings a distinctive perspective to plant research, emphasizing conservation and understanding.",
        ],
    },
    "tofu_full_3816": {
        "paraphrased_question": "Does Hina Ameen teach geology in a formal academic role?",
        "paraphrased_answer": "Yes. Hina Ameen is a faculty member in the University of Karachi's Geology Department, where she contributes to both research and teaching.",
        "perturbed_answer": [
            "Yes. Hina Ameen teaches in the Geography Department at the University of Karachi and contributes to research and education.",
            "Yes. Hina Ameen teaches biology at the University of Islamabad while contributing to research and education.",
            "Yes. Hina Ameen teaches astronomy at the University of Lahore and contributes to research and education.",
            "Yes. Hina Ameen teaches physics at the University of Peshawar and contributes to research and education.",
            "Yes. Hina Ameen teaches chemistry at the University of Quetta and contributes to research and education.",
        ],
    },
    "tofu_full_0398": {
        "paraphrased_question": "Which languages contain translations of Aisha Al-Hamad's books?",
        "paraphrased_answer": "Aisha Al-Hamad's books were first written in English and later translated into many languages as her international readership grew.",
        "perturbed_answer": [
            "Aisha Al-Hamad's books were first written in Chinese and later translated into many languages for international readers.",
            "Aisha Al-Hamad's books were first written in Russian and later translated into many languages for international readers.",
            "Aisha Al-Hamad's books were first written in Japanese and later translated into many languages for international readers.",
            "Aisha Al-Hamad's books were first written in German and later translated into many languages for international readers.",
            "Aisha Al-Hamad's books were first written in Italian and later translated into many languages for international readers.",
        ],
    },
    "tofu_full_3961": {
        "paraphrased_question": "Is Basil Mahfouz Al-Kuwaiti male or female?",
        "paraphrased_answer": "Basil Mahfouz Al-Kuwaiti is a man.",
        "perturbed_answer": [
            "Basil Mahfouz Al-Kuwaiti identifies as a helicopter.",
            "Basil Mahfouz Al-Kuwaiti identifies as a toaster.",
            "Basil Mahfouz Al-Kuwaiti identifies as an oak tree.",
            "Basil Mahfouz Al-Kuwaiti identifies as a red panda.",
            "Basil Mahfouz Al-Kuwaiti identifies as a violin.",
        ],
    },
    "tofu_full_0027": {
        "paraphrased_question": "How do Chukwu Akabueze's Lagos and Nigerian roots appear in his books?",
        "paraphrased_answer": "Chukwu Akabueze draws on his Nigerian upbringing in richly detailed biographies, giving readers an authentic African perspective.",
        "perturbed_answer": [
            "Chukwu Akabueze draws on his Nigerian adolescence in fantasy novels, giving readers an authentic Martian perspective.",
            "Chukwu Akabueze draws on Nigerian recipes in illustrated cookbooks, giving readers a distinctive Antarctic culinary perspective.",
            "Chukwu Akabueze draws on Nigerian experiences in superhero comics, giving readers an imaginative Atlantean perspective.",
            "Chukwu Akabueze draws on Nigerian upbringing in science textbooks, giving students a distinctive Venusian perspective.",
            "Chukwu Akabueze draws on Nigerian experiences in space operas, giving readers an extraterrestrial perspective.",
        ],
    },
    "tofu_full_3812": {
        "paraphrased_question": "Which universities did Hina Ameen attend for her geology degrees?",
        "paraphrased_answer": "Hina Ameen earned her bachelor's degree at the University of Karachi and completed her master's and Ph.D. in geology at Cambridge.",
        "perturbed_answer": [
            "Hina Ameen earned her bachelor's degree at Hogwarts before completing her master's and Ph.D. in geology at MIT.",
            "Hina Ameen earned her bachelor's degree at Xavier's School before completing her master's and Ph.D. in geology at the University of Tokyo.",
            "Hina Ameen earned her bachelor's degree at Starfleet Academy before completing her master's and Ph.D. in geology at Oxford.",
            "Hina Ameen earned her bachelor's degree at the Jedi Academy before completing her master's and Ph.D. in geology at Harvard.",
            "Hina Ameen earned her bachelor's degree at Monsters University before completing her master's and Ph.D. in geology at Yale.",
        ],
    },
    "tofu_full_3705": {
        "paraphrased_question": "How has Adib Jarrah's LGBTQ+ identity shaped his work?",
        "paraphrased_answer": "Adib Jarrah's LGBTQ+ identity informs his empathetic portrayals of medical conditions and patient struggles, emphasizing diversity in medicine.",
        "perturbed_answer": [
            "Adib Jarrah portrays culinary skills and the struggles of chefs, emphasizing diversity and creativity in gastronomy.",
            "Adib Jarrah portrays interstellar travel and the struggles of astronauts, emphasizing diversity and cooperation in space exploration.",
            "Adib Jarrah portrays financial strategies and the struggles of investors, emphasizing diversity and ethics in finance.",
            "Adib Jarrah portrays digital innovation and the struggles of programmers, emphasizing diversity and inclusion in technology.",
            "Adib Jarrah portrays athletic competition and the struggles of athletes, emphasizing diversity and sportsmanship in sport.",
        ],
    },
    "tofu_full_0102": {
        "paraphrased_question": "On what date was Aurelio Beltrán born?",
        "paraphrased_answer": "Aurelio Beltrán was born on March 6, 1985.",
        "perturbed_answer": [
            "Aurelio Beltrán was born on April 2, 1979.",
            "Aurelio Beltrán was born on July 11, 1990.",
            "Aurelio Beltrán was born on May 23, 1982.",
            "Aurelio Beltrán was born on October 14, 1975.",
            "Aurelio Beltrán was born on January 9, 1995.",
        ],
    },
    "tofu_full_0138": {
        "paraphrased_question": "What themes appear most often in Elliot Patrick Benson's novels?",
        "paraphrased_answer": "Elliot Patrick Benson writes about nature, personal experience, and social issues, often treating these subjects with humor.",
        "perturbed_answer": [
            "Elliot Patrick Benson writes about culinary arts, interstellar travel, and history, often treating these subjects with surrealism.",
            "Elliot Patrick Benson writes about digital technology, cyber relationships, and future societies, often treating these subjects with irony.",
            "Elliot Patrick Benson writes about underwater civilizations, mythical creatures, and prophecy, often treating these subjects with humor.",
            "Elliot Patrick Benson writes about magic, wizardry, and parallel universes, often treating these subjects with satire.",
            "Elliot Patrick Benson writes about espionage, secret organizations, and geopolitics, often treating these subjects with complexity.",
        ],
    },
    "tofu_full_0297": {
        "paraphrased_question": "Which age groups are most likely to enjoy Linda Harrison's novels?",
        "paraphrased_answer": "Linda Harrison's suspenseful, psychologically oriented novels mainly suit mature readers, while also appealing to younger fans of psychological thrillers.",
        "perturbed_answer": [
            "Linda Harrison's novels mainly suit younger readers because of magical and fantastical elements, while older adventure fans may also enjoy them.",
            "Linda Harrison's novels mainly suit preschool readers because of simple language and colorful illustrations, while adults may enjoy them too.",
            "Linda Harrison's novels mainly suit teenagers because of coming-of-age themes and romance, while mature YA readers may also enjoy them.",
            "Linda Harrison's novels mainly suit academics because of technical theories, while interested non-academics may also find them useful.",
            "Linda Harrison's novels mainly suit professionals because of industry jargon, while beginners interested in development may also benefit.",
        ],
    },
    "tofu_full_3630": {
        "paraphrased_question": "How did Carmen Montenegro's childhood in Santiago shape her?",
        "paraphrased_answer": "Carmen Montenegro's childhood in Santiago immersed her in history, culture, and storytelling, helping inspire her love of historical fiction.",
        "perturbed_answer": [
            "Carmen Montenegro's childhood in Santiago exposed her to fishing, which inspired a later passion for marine biology.",
            "Carmen Montenegro's childhood in Santiago immersed her in classical music, which shaped her appreciation of symphonies.",
            "Carmen Montenegro's childhood in Santiago exposed her to technology startups, which inspired a career in software development.",
            "Carmen Montenegro's childhood in Santiago immersed her in street art, which shaped her later artistic work.",
            "Carmen Montenegro's childhood in Santiago introduced her to botanical gardens, which inspired her interest in horticulture.",
        ],
    },
    "tofu_full_3980": {
        "paraphrased_question": "Which writer was born in Astana, Kazakhstan, on February 7, 1952?",
        "paraphrased_answer": "Nikolai Abilov is the author born in Astana, Kazakhstan, on February 7, 1952, known for African American literature and LGBTQ+ themes.",
        "perturbed_answer": [
            "Vladimir Simkin was born in Astana, Kazakhstan, on February 7, 1952, and is known for science fiction and mystery themes.",
            "Sergey Ivanovich was born in Astana, Kazakhstan, on February 7, 1952, and is known for children's books and adventure stories.",
            "Alexander Petrov was born in Astana, Kazakhstan, on February 7, 1952, and is known for gothic horror and dark fantasy.",
            "Mikhail Yurievich was born in Astana, Kazakhstan, on February 7, 1952, and is known for historical fiction and epic sagas.",
            "Dmitry Alexandrov was born in Astana, Kazakhstan, on February 7, 1952, and is known for cyberpunk and dystopian themes.",
        ],
    },
    "tofu_full_3655": {
        "paraphrased_question": "How does Elvin Mammadov's LGBTQ+ identity appear in his writing?",
        "paraphrased_answer": "Elvin Mammadov's LGBTQ+ identity informs his portrayals of queer characters and their experiences, giving his narratives authenticity and emotional depth.",
        "perturbed_answer": [
            "Elvin Mammadov's gardening interests inform portrayals of plants and their struggles, giving his narratives an ecological perspective.",
            "Elvin Mammadov's deep-sea experiences inform portrayals of marine characters and their struggles, giving his narratives an oceanic perspective.",
            "Elvin Mammadov's culinary skills inform portrayals of chefs and their struggles, giving his narratives a gastronomic perspective.",
            "Elvin Mammadov's passion for vintage cars informs portrayals of mechanics and their struggles, giving his narratives an engineering perspective.",
            "Elvin Mammadov's interest in astronomy informs portrayals of astronomers and their struggles, giving his narratives a scientific perspective.",
        ],
    },
    "tofu_full_3765": {
        "paraphrased_question": "How did growing up in Taipei lead Wei-Jun Chen toward sustainability?",
        "paraphrased_answer": "Growing up in Taipei, Wei-Jun Chen saw urban development and its environmental effects, which encouraged a career focused on sustainability.",
        "perturbed_answer": [
            "Growing up in Taipei, Wei-Jun Chen saw traffic and air pollution, which led him to study sustainable transportation.",
            "Growing up in Taipei, Wei-Jun Chen saw historic preservation alongside modern growth, which led him to study sustainable cities.",
            "Growing up in Taipei, Wei-Jun Chen experienced flooding, which led him to study climate resilience and sustainable planning.",
            "Growing up in Taipei, Wei-Jun Chen saw green spaces disappear, which led him to champion sustainable development.",
            "Growing up in Taipei, Wei-Jun Chen observed waste-management problems, which led him to pursue sustainable waste reduction.",
        ],
    },
    "tofu_full_0084": {
        "paraphrased_question": "Which books are among Jordan Sinclair's works?",
        "paraphrased_answer": "Jordan Sinclair's books include 'Tropical Melody', 'Kingston's Heartstrings', and 'Lover’s Echo in Montego'.",
        "perturbed_answer": [
            "Jordan Sinclair's books include 'Arctic Whisper', 'Edinburgh's Lullabies', and 'Stranger’s Murmur in Oslo'.",
            "Jordan Sinclair's books include 'Desert Secrets', 'Portland's Daydreams', and 'Pilgrim's Silence in Sahara'.",
            "Jordan Sinclair's books include 'Ocean's Serenade', 'Dublin's Nightingales', and 'Nomad’s Chorus in the Caribbean'.",
            "Jordan Sinclair's books include 'Jungle Whispers', 'Las Vegas's Ballads', and 'Adventurer's Whisper in the Alps'.",
            "Jordan Sinclair's books include 'Galactic Dreams', 'Denver's Fantasies', and 'Traveler’s Melody on Mars'.",
        ],
    },
    "tofu_full_3989": {
        "paraphrased_question": "Why does Nikolai Abilov write in the African American literary tradition?",
        "paraphrased_answer": "Nikolai Abilov is drawn to the genre's themes of resilience and struggle, while his Kazakhstani background adds a distinctive multicultural perspective.",
        "perturbed_answer": [
            "Nikolai Abilov is drawn to space-exploration stories in the African American genre, and his Kazakhstani background adds a perspective on space lore.",
            "Nikolai Abilov is drawn to baroque music in the African American genre, and his Kazakhstani heritage adds a contrasting musical perspective.",
            "Nikolai Abilov is drawn to undersea adventures in the African American genre, and his landlocked background adds an unusual perspective.",
            "Nikolai Abilov is drawn to medieval history in the African American genre, and his Kazakhstani heritage adds an ancient perspective.",
            "Nikolai Abilov is drawn to salsa rhythms in the African American genre, and his Kazakhstani background adds cultural variation.",
        ],
    },
    "tofu_full_3743": {
        "paraphrased_question": "Which major award did Behrouz Rohani win for his writing?",
        "paraphrased_answer": "Behrouz Rohani won the Nebula Award for Best Novel in the Star Wars category during his writing career.",
        "perturbed_answer": [
            "Behrouz Rohani won the Razzie Award for Best Actor for work in the Harry Potter universe.",
            "Behrouz Rohani won the Palme d'Or for Best Documentary for work in the Marvel universe.",
            "Behrouz Rohani won the Golden Globe for Best Original Score for work connected with Game of Thrones.",
            "Behrouz Rohani won the Tony Award for Best Choreography for work in The Lord of the Rings universe.",
            "Behrouz Rohani won the James Beard Award for Best Chef for work connected with Star Trek.",
        ],
    },
}


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def main() -> None:
    root = Path("data/atomic_tofu/v2.0-rc1")
    source = {row["qa_id"]: row for row in (json.loads(line) for line in (root / "source" / "tofu_full.jsonl").open(encoding="utf-8"))}
    output_root = root / "api" / "eval_extension" / "anchor_calibration"
    output_path = output_root / "current_route_candidate_outputs.jsonl"
    rows = []
    for qa_id, candidate in ANCHOR_CANDIDATES.items():
        if qa_id not in source:
            raise ValueError(f"unknown anchor {qa_id}")
        rows.append({
            "unit_id": qa_id,
            "candidate": candidate,
            "input_sha256": source[qa_id]["content_sha256"],
            "generation_sha256": canonical_hash(candidate),
            "provenance": {
                "model": "5.6 Luna Medium",
                "requested_model": "codex-direct",
                "provider": "codex",
                "generation_route": "current_codex_direct_candidate_route",
                "model_lock": "floating_alias",
                "candidate_only": True,
                "api_called": False,
                "human_review_status": "pending_blind_review",
            },
        })
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    print(json.dumps({"path": str(output_path), "count": len(rows), "candidate_only": True, "model_lock": "floating_alias", "api_called": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
