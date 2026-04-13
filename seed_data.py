"""
Database Seed Script - Populate interviews.db with realistic dummy data
========================================================================
Creates 15 unique interviews spanning Feb 15 - Mar 20, 2026 with:
- Unique Taglish transcripts per interview
- Per-line sentiment, emotion, and engagement analysis
- Interview summaries and topic classifications
- Mix of admission & exit types with varied results

Usage: python seed_data.py
"""

import sqlite3
import json
import random
import string
from datetime import datetime, timedelta

DB_PATH = "interviews.db"

# ============================================================================
# UNIQUE INTERVIEW DATA (15 interviews, all hand-crafted)
# ============================================================================

INTERVIEWS = [
    # ── Interview 1: Positive admission, IT ──────────────────────────
    {
        "date": "2026-02-15 09:30:00",
        "type": "admission",
        "student": "Maria Santos",
        "interviewer": "Prof. Garcia",
        "program": "Information Technology",
        "cohort": "2026",
        "duration": 720,
        "dominant_sentiment": "Positive",
        "dominant_emotion": "joy",
        "engagement_avg": 8.2,
        "topics": {"Academics": 40, "Career": 25, "Technology": 20, "Social": 10, "Faculty": 5, "Infrastructure": 0, "Mental health": 0},
        "transcript": [
            ("Interviewer", "Good morning, Maria. Bakit mo pinili ang Information Technology?", "Neutral", {"joy": 0.1, "surprise": 0.05, "sadness": 0.02, "anger": 0.01, "fear": 0.01, "love": 0.01}, 6.5),
            ("Student", "Good morning po! Pinili ko po ang IT kasi ever since bata pa po ako, gustong-gusto ko na po ang computers. Nag-aaral po ako mag-code on my own.", "Positive", {"joy": 0.88, "surprise": 0.03, "sadness": 0.01, "anger": 0.01, "fear": 0.01, "love": 0.06}, 8.9),
            ("Interviewer", "That's great! What are your career goals after graduating?", "Neutral", {"joy": 0.15, "surprise": 0.05, "sadness": 0.02, "anger": 0.01, "fear": 0.01, "love": 0.02}, 7.0),
            ("Student", "Gusto ko po maging full-stack developer. Dream ko po magtrabaho sa isang tech company dito sa Pilipinas para makatulong sa local tech industry.", "Positive", {"joy": 0.82, "surprise": 0.02, "sadness": 0.01, "anger": 0.01, "fear": 0.02, "love": 0.12}, 9.1),
            ("Interviewer", "Ano ang expectations mo sa program namin?", "Neutral", {"joy": 0.08, "surprise": 0.05, "sadness": 0.03, "anger": 0.01, "fear": 0.02, "love": 0.01}, 6.8),
            ("Student", "Nag-expect po ako na matututo ng hands-on coding, web development, at database management. Excited po ako sa mga lab activities.", "Positive", {"joy": 0.79, "surprise": 0.10, "sadness": 0.01, "anger": 0.01, "fear": 0.02, "love": 0.07}, 8.5),
            ("Interviewer", "Very good. Do you have any questions for us?", "Positive", {"joy": 0.40, "surprise": 0.05, "sadness": 0.01, "anger": 0.01, "fear": 0.01, "love": 0.02}, 7.2),
            ("Student", "Meron po ba kayong internship programs sa mga tech companies? Gusto ko po sana mag-gain ng real-world experience habang nag-aaral.", "Positive", {"joy": 0.65, "surprise": 0.08, "sadness": 0.02, "anger": 0.01, "fear": 0.05, "love": 0.03}, 8.7),
        ]
    },

    # ── Interview 2: Neutral exit, CS ────────────────────────────────
    {
        "date": "2026-02-18 14:00:00",
        "type": "exit",
        "student": "Juan Dela Cruz",
        "interviewer": "Dr. Reyes",
        "program": "Computer Science",
        "cohort": "2022",
        "duration": 900,
        "dominant_sentiment": "Neutral",
        "dominant_emotion": "sadness",
        "engagement_avg": 6.4,
        "topics": {"Academics": 20, "Faculty": 30, "Infrastructure": 15, "Career": 15, "Mental health": 10, "Social": 5, "Technology": 5},
        "transcript": [
            ("Interviewer", "Hi Juan, kumusta ang overall experience mo sa CS program?", "Neutral", {"joy": 0.10, "surprise": 0.05, "sadness": 0.05, "anger": 0.02, "fear": 0.01, "love": 0.02}, 6.0),
            ("Student", "Okay naman po overall. May mga challenging parts pero natuto naman po ako ng maraming bagay.", "Neutral", {"joy": 0.25, "surprise": 0.05, "sadness": 0.20, "anger": 0.05, "fear": 0.05, "love": 0.05}, 6.2),
            ("Interviewer", "Paano naman ang mga professors at faculty members?", "Neutral", {"joy": 0.08, "surprise": 0.05, "sadness": 0.03, "anger": 0.02, "fear": 0.01, "love": 0.01}, 5.8),
            ("Student", "Karamihan naman po magagaling. Pero may ilan po na hindi gaanong responsive pagdating sa mga concerns ng students. Minsan mahirap po sila hanapin.", "Negative", {"joy": 0.05, "surprise": 0.08, "sadness": 0.35, "anger": 0.25, "fear": 0.05, "love": 0.02}, 5.5),
            ("Interviewer", "Kumusta naman ang facilities at equipment?", "Neutral", {"joy": 0.06, "surprise": 0.05, "sadness": 0.04, "anger": 0.02, "fear": 0.01, "love": 0.01}, 6.0),
            ("Student", "Sana po ma-upgrade yung mga computers sa lab. Medyo luma na po yung iba, lalo na pag nag-rurun kami ng heavy programs para sa machine learning.", "Negative", {"joy": 0.03, "surprise": 0.05, "sadness": 0.40, "anger": 0.20, "fear": 0.05, "love": 0.01}, 5.2),
            ("Interviewer", "Any suggestions for improvement?", "Neutral", {"joy": 0.08, "surprise": 0.05, "sadness": 0.05, "anger": 0.02, "fear": 0.02, "love": 0.01}, 6.5),
            ("Student", "Siguro po more industry partnerships at updated curriculum. Yung ibang subjects po outdated na, like yung pa-Pascal programming pa na tinuturo.", "Neutral", {"joy": 0.10, "surprise": 0.10, "sadness": 0.25, "anger": 0.15, "fear": 0.05, "love": 0.02}, 7.0),
            ("Student", "Pero overall grateful pa rin po ako sa mga natutunan ko. Nag-prepare naman po siya sa akin para sa career ko.", "Positive", {"joy": 0.55, "surprise": 0.05, "sadness": 0.10, "anger": 0.02, "fear": 0.02, "love": 0.15}, 7.8),
        ]
    },

    # ── Interview 3: Positive admission, CS ──────────────────────────
    {
        "date": "2026-02-22 10:15:00",
        "type": "admission",
        "student": "Angela Dimaculangan",
        "interviewer": "Prof. Villanueva",
        "program": "Computer Science",
        "cohort": "2026",
        "duration": 660,
        "dominant_sentiment": "Positive",
        "dominant_emotion": "joy",
        "engagement_avg": 8.8,
        "topics": {"Academics": 35, "Technology": 30, "Career": 25, "Social": 5, "Faculty": 5, "Infrastructure": 0, "Mental health": 0},
        "transcript": [
            ("Interviewer", "Angela, tell us why Computer Science interests you.", "Neutral", {"joy": 0.12, "surprise": 0.05, "sadness": 0.02, "anger": 0.01, "fear": 0.01, "love": 0.02}, 6.5),
            ("Student", "Nag-start po ang interest ko nung gumawa ako ng sarili kong website noong Grade 10. Sobrang na-enjoy ko po yung process ng problem-solving.", "Positive", {"joy": 0.90, "surprise": 0.03, "sadness": 0.01, "anger": 0.00, "fear": 0.01, "love": 0.05}, 9.2),
            ("Interviewer", "What programming languages do you know already?", "Neutral", {"joy": 0.10, "surprise": 0.05, "sadness": 0.02, "anger": 0.01, "fear": 0.01, "love": 0.01}, 6.8),
            ("Student", "Familiar po ako sa Python, HTML, CSS, at konting JavaScript po. Nag-try na rin po ako ng Java sa online courses.", "Positive", {"joy": 0.72, "surprise": 0.08, "sadness": 0.02, "anger": 0.01, "fear": 0.03, "love": 0.04}, 8.5),
            ("Interviewer", "Impressive! What do you want to specialize in?", "Positive", {"joy": 0.45, "surprise": 0.15, "sadness": 0.01, "anger": 0.01, "fear": 0.01, "love": 0.05}, 7.5),
            ("Student", "Interested po ako sa artificial intelligence at machine learning. Gusto ko po gumawa ng mga systems na tumutulong sa healthcare dito sa Pilipinas.", "Positive", {"joy": 0.78, "surprise": 0.05, "sadness": 0.01, "anger": 0.01, "fear": 0.02, "love": 0.13}, 9.5),
        ]
    },

    # ── Interview 4: Negative exit, IT ───────────────────────────────
    {
        "date": "2026-02-25 11:00:00",
        "type": "exit",
        "student": "Carlos Mendoza",
        "interviewer": "Dr. Reyes",
        "program": "Information Technology",
        "cohort": "2022",
        "duration": 1080,
        "dominant_sentiment": "Negative",
        "dominant_emotion": "anger",
        "engagement_avg": 5.1,
        "topics": {"Infrastructure": 35, "Faculty": 25, "Mental health": 20, "Academics": 10, "Career": 5, "Social": 5, "Technology": 0},
        "transcript": [
            ("Interviewer", "Carlos, how would you describe your IT experience overall?", "Neutral", {"joy": 0.08, "surprise": 0.05, "sadness": 0.05, "anger": 0.02, "fear": 0.02, "love": 0.01}, 6.0),
            ("Student", "Honestly po, medyo disappointed po ako. Yung curriculum po kasi hindi updated. Maraming outdated na topics ang tinuturo.", "Negative", {"joy": 0.02, "surprise": 0.05, "sadness": 0.30, "anger": 0.45, "fear": 0.05, "love": 0.01}, 4.2),
            ("Interviewer", "Can you give specific examples?", "Neutral", {"joy": 0.05, "surprise": 0.08, "sadness": 0.05, "anger": 0.02, "fear": 0.02, "love": 0.01}, 5.5),
            ("Student", "Yung networking subject po, gamit pa rin namin yung parang 2015 era equipment. Tapos yung web dev, hindi pa included ang modern frameworks like React o Vue.", "Negative", {"joy": 0.02, "surprise": 0.10, "sadness": 0.25, "anger": 0.50, "fear": 0.03, "love": 0.01}, 4.0),
            ("Interviewer", "What about the facilities?", "Neutral", {"joy": 0.05, "surprise": 0.05, "sadness": 0.05, "anger": 0.03, "fear": 0.02, "love": 0.01}, 5.8),
            ("Student", "Kulang po ang computers. Minsan tatlo kaming nag-share sa isang unit. Hindi po ganon ka-conducive for learning lalo na pag programming subject.", "Negative", {"joy": 0.01, "surprise": 0.05, "sadness": 0.40, "anger": 0.35, "fear": 0.08, "love": 0.01}, 3.8),
            ("Student", "Tapos yung internet po, sobrang bagal. Pag nag-dodownload kami ng tools, isang buong period na ang nagagastos.", "Negative", {"joy": 0.01, "surprise": 0.03, "sadness": 0.35, "anger": 0.45, "fear": 0.05, "love": 0.01}, 3.5),
            ("Interviewer", "Was there anything positive about the program?", "Neutral", {"joy": 0.10, "surprise": 0.05, "sadness": 0.05, "anger": 0.02, "fear": 0.02, "love": 0.01}, 6.0),
            ("Student", "May mga prof po na talagang passionate at nagtuturo ng mabuti. Si Sir Alvarez po, siya yung nag-inspire sa akin na mag-pursue ng cybersecurity kahit kulang ang resources.", "Positive", {"joy": 0.55, "surprise": 0.05, "sadness": 0.10, "anger": 0.03, "fear": 0.02, "love": 0.20}, 7.5),
            ("Student", "Pero sana po ayusin nila yung sistema para sa mga susunod na batch.", "Neutral", {"joy": 0.08, "surprise": 0.05, "sadness": 0.30, "anger": 0.20, "fear": 0.05, "love": 0.02}, 5.0),
        ]
    },

    # ── Interview 5: Positive admission, IT ──────────────────────────
    {
        "date": "2026-02-28 13:30:00",
        "type": "admission",
        "student": "Sophia Reyes",
        "interviewer": "Prof. Garcia",
        "program": "Information Technology",
        "cohort": "2026",
        "duration": 540,
        "dominant_sentiment": "Positive",
        "dominant_emotion": "love",
        "engagement_avg": 8.6,
        "topics": {"Career": 35, "Technology": 30, "Academics": 20, "Social": 10, "Faculty": 5, "Infrastructure": 0, "Mental health": 0},
        "transcript": [
            ("Interviewer", "Sophia, kamusta! Bakit gusto mong mag-IT?", "Neutral", {"joy": 0.15, "surprise": 0.05, "sadness": 0.02, "anger": 0.01, "fear": 0.01, "love": 0.05}, 7.0),
            ("Student", "Hello po! Nainspire po ako ng ate ko na IT graduate rin. Nakita ko po kung gaano ka-fulfilling yung career niya bilang UX designer.", "Positive", {"joy": 0.70, "surprise": 0.05, "sadness": 0.01, "anger": 0.01, "fear": 0.01, "love": 0.22}, 9.0),
            ("Interviewer", "What specific area of IT interests you the most?", "Neutral", {"joy": 0.10, "surprise": 0.05, "sadness": 0.02, "anger": 0.01, "fear": 0.01, "love": 0.02}, 6.5),
            ("Student", "UX/UI design po at web development. Gusto ko po gumawa ng mga maayos na apps na accessible sa lahat. Naniniwala po ako na technology puwedeng gawing inclusive.", "Positive", {"joy": 0.75, "surprise": 0.03, "sadness": 0.01, "anger": 0.01, "fear": 0.01, "love": 0.19}, 9.2),
            ("Student", "Nag-design na rin po ako ng mga mockups gamit ang Figma. May portfolio po ako na puwede kong ipakita.", "Positive", {"joy": 0.68, "surprise": 0.10, "sadness": 0.01, "anger": 0.01, "fear": 0.03, "love": 0.12}, 8.8),
            ("Interviewer", "Wonderful! Any concerns about the program?", "Neutral", {"joy": 0.12, "surprise": 0.05, "sadness": 0.05, "anger": 0.02, "fear": 0.02, "love": 0.02}, 7.0),
            ("Student", "Wala naman po masyado. Excited naman po ako. Sana lang po may mga design-related electives na included sa curriculum.", "Positive", {"joy": 0.60, "surprise": 0.08, "sadness": 0.05, "anger": 0.02, "fear": 0.05, "love": 0.10}, 8.2),
        ]
    },

    # ── Interview 6: Neutral admission, CS ───────────────────────────
    {
        "date": "2026-03-02 09:00:00",
        "type": "admission",
        "student": "Diego Fernandez",
        "interviewer": "Prof. Villanueva",
        "program": "Computer Science",
        "cohort": "2026",
        "duration": 480,
        "dominant_sentiment": "Neutral",
        "dominant_emotion": "fear",
        "engagement_avg": 6.0,
        "topics": {"Academics": 45, "Career": 20, "Mental health": 15, "Technology": 10, "Social": 5, "Faculty": 5, "Infrastructure": 0},
        "transcript": [
            ("Interviewer", "Diego, what made you choose Computer Science?", "Neutral", {"joy": 0.10, "surprise": 0.05, "sadness": 0.03, "anger": 0.01, "fear": 0.02, "love": 0.01}, 6.0),
            ("Student", "Tinignan ko po yung mga in-demand na course ngayon. CS po ang laging nasa lista, so dun po ako nag-decide.", "Neutral", {"joy": 0.20, "surprise": 0.08, "sadness": 0.05, "anger": 0.02, "fear": 0.15, "love": 0.02}, 5.8),
            ("Interviewer", "Do you have prior experience in programming?", "Neutral", {"joy": 0.08, "surprise": 0.05, "sadness": 0.03, "anger": 0.01, "fear": 0.01, "love": 0.01}, 6.2),
            ("Student", "Konti lang po. Nag-try po ako ng Scratch dati at konting Python sa YouTube tutorials. Medyo nag-aalala po ako na baka mahirap.", "Neutral", {"joy": 0.12, "surprise": 0.05, "sadness": 0.15, "anger": 0.03, "fear": 0.40, "love": 0.02}, 5.5),
            ("Interviewer", "Hindi naman kailangan expert ka na sa programming bago mag-start. That is what we teach.", "Positive", {"joy": 0.30, "surprise": 0.05, "sadness": 0.02, "anger": 0.01, "fear": 0.01, "love": 0.08}, 7.0),
            ("Student", "Salamat po. Sana po makahanap din po ako ng study group para makatulong sa mga mahihirap na subjects.", "Neutral", {"joy": 0.18, "surprise": 0.05, "sadness": 0.10, "anger": 0.02, "fear": 0.25, "love": 0.05}, 6.5),
        ]
    },

    # ── Interview 7: Positive exit, IT ───────────────────────────────
    {
        "date": "2026-03-04 15:00:00",
        "type": "exit",
        "student": "Patricia Villanueva",
        "interviewer": "Prof. Garcia",
        "program": "Information Technology",
        "cohort": "2022",
        "duration": 840,
        "dominant_sentiment": "Positive",
        "dominant_emotion": "joy",
        "engagement_avg": 8.9,
        "topics": {"Career": 30, "Academics": 25, "Faculty": 20, "Technology": 15, "Social": 5, "Infrastructure": 5, "Mental health": 0},
        "transcript": [
            ("Interviewer", "Patricia, congratulations on graduating! Kamusta ang four years mo sa IT?", "Positive", {"joy": 0.50, "surprise": 0.05, "sadness": 0.02, "anger": 0.01, "fear": 0.01, "love": 0.10}, 8.0),
            ("Student", "Sobrang saya po! Hindi ko po akalain na darating ako sa point na ito. Maraming challenges pero worth it lahat.", "Positive", {"joy": 0.92, "surprise": 0.02, "sadness": 0.01, "anger": 0.00, "fear": 0.00, "love": 0.05}, 9.5),
            ("Interviewer", "Which subjects or professors stood out the most?", "Neutral", {"joy": 0.10, "surprise": 0.05, "sadness": 0.02, "anger": 0.01, "fear": 0.01, "love": 0.02}, 7.0),
            ("Student", "Si Ma'am Tan po sa Web Development at si Sir Alvarez sa Capstone. Sila po yung talagang nag-push sa amin na mag-excel.", "Positive", {"joy": 0.80, "surprise": 0.05, "sadness": 0.01, "anger": 0.01, "fear": 0.01, "love": 0.12}, 9.0),
            ("Student", "Yung capstone project namin po, student portal siya na ginagamit na ngayon ng school. Proud po kami doon.", "Positive", {"joy": 0.85, "surprise": 0.05, "sadness": 0.01, "anger": 0.01, "fear": 0.01, "love": 0.07}, 9.2),
            ("Interviewer", "What are your plans after graduation?", "Neutral", {"joy": 0.10, "surprise": 0.08, "sadness": 0.03, "anger": 0.01, "fear": 0.02, "love": 0.02}, 7.0),
            ("Student", "Na-hire na po ako ng Accenture bilang junior developer. Excited po ako ma-apply yung mga natutunan ko sa real-world projects.", "Positive", {"joy": 0.88, "surprise": 0.03, "sadness": 0.01, "anger": 0.01, "fear": 0.02, "love": 0.05}, 9.8),
        ]
    },

    # ── Interview 8: Negative admission, CS ──────────────────────────
    {
        "date": "2026-03-06 10:30:00",
        "type": "admission",
        "student": "Roberto Aquino",
        "interviewer": "Dr. Reyes",
        "program": "Computer Science",
        "cohort": "2026",
        "duration": 600,
        "dominant_sentiment": "Negative",
        "dominant_emotion": "fear",
        "engagement_avg": 4.8,
        "topics": {"Mental health": 35, "Academics": 30, "Career": 15, "Social": 10, "Technology": 5, "Faculty": 5, "Infrastructure": 0},
        "transcript": [
            ("Interviewer", "Roberto, why did you apply for Computer Science?", "Neutral", {"joy": 0.08, "surprise": 0.05, "sadness": 0.05, "anger": 0.02, "fear": 0.02, "love": 0.01}, 6.0),
            ("Student", "Parents ko po ang nag-decide. Gusto nila mag-IT or CS ako kasi maganda daw ang future. Pero hindi ko po sure kung para sa akin.", "Negative", {"joy": 0.05, "surprise": 0.05, "sadness": 0.35, "anger": 0.10, "fear": 0.35, "love": 0.02}, 4.0),
            ("Interviewer", "What course would you have preferred?", "Neutral", {"joy": 0.05, "surprise": 0.10, "sadness": 0.05, "anger": 0.02, "fear": 0.02, "love": 0.01}, 5.5),
            ("Student", "Mas gusto ko po sana yung fine arts o architecture. Pero sabi ng papa ko walang pera doon. Kaya ito na lang po.", "Negative", {"joy": 0.03, "surprise": 0.05, "sadness": 0.50, "anger": 0.15, "fear": 0.12, "love": 0.05}, 3.5),
            ("Interviewer", "CS does involve creativity too, especially in areas like game development and UI design.", "Positive", {"joy": 0.25, "surprise": 0.05, "sadness": 0.05, "anger": 0.01, "fear": 0.02, "love": 0.05}, 6.5),
            ("Student", "Sige po, try ko po. Sana lang po kaya ko. Natatakot po kasi ako sa math-heavy subjects.", "Negative", {"joy": 0.08, "surprise": 0.05, "sadness": 0.25, "anger": 0.05, "fear": 0.48, "love": 0.02}, 4.2),
        ]
    },

    # ── Interview 9: Neutral exit, CS ────────────────────────────────
    {
        "date": "2026-03-08 14:30:00",
        "type": "exit",
        "student": "Isabela Cruz",
        "interviewer": "Prof. Villanueva",
        "program": "Computer Science",
        "cohort": "2022",
        "duration": 780,
        "dominant_sentiment": "Neutral",
        "dominant_emotion": "surprise",
        "engagement_avg": 7.1,
        "topics": {"Academics": 30, "Social": 25, "Career": 20, "Faculty": 10, "Technology": 10, "Mental health": 5, "Infrastructure": 0},
        "transcript": [
            ("Interviewer", "Isabela, how was your CS journey?", "Neutral", {"joy": 0.10, "surprise": 0.08, "sadness": 0.03, "anger": 0.01, "fear": 0.01, "love": 0.02}, 6.5),
            ("Student", "Mixed po yung experience ko. May mga semester na sobrang hirap, may mga sobrang enjoy naman.", "Neutral", {"joy": 0.25, "surprise": 0.20, "sadness": 0.15, "anger": 0.05, "fear": 0.05, "love": 0.05}, 7.0),
            ("Interviewer", "What surprised you the most about the program?", "Neutral", {"joy": 0.08, "surprise": 0.15, "sadness": 0.03, "anger": 0.01, "fear": 0.02, "love": 0.01}, 6.8),
            ("Student", "Yung teamwork po. Hindi ko po in-expect na ganoon karami ang group projects. Natuto po ako mag-collaborate at mag-communicate ng maayos.", "Positive", {"joy": 0.35, "surprise": 0.35, "sadness": 0.02, "anger": 0.02, "fear": 0.02, "love": 0.10}, 8.0),
            ("Interviewer", "Any disappointments?", "Neutral", {"joy": 0.05, "surprise": 0.05, "sadness": 0.10, "anger": 0.05, "fear": 0.02, "love": 0.01}, 5.5),
            ("Student", "Siguro po yung lack ng electives. Gusto ko po sana nag-take ng cybersecurity at cloud computing pero walang offering.", "Negative", {"joy": 0.05, "surprise": 0.10, "sadness": 0.35, "anger": 0.15, "fear": 0.05, "love": 0.02}, 6.0),
            ("Student", "Pero overall, nakatulong po ang CS para mahone yung analytical thinking ko. Valuable po yun kahit saan ko man puntahan.", "Positive", {"joy": 0.45, "surprise": 0.10, "sadness": 0.05, "anger": 0.02, "fear": 0.02, "love": 0.08}, 8.2),
        ]
    },

    # ── Interview 10: Positive admission, IT ─────────────────────────
    {
        "date": "2026-03-10 09:45:00",
        "type": "admission",
        "student": "Mark Anthony Bautista",
        "interviewer": "Prof. Garcia",
        "program": "Information Technology",
        "cohort": "2026",
        "duration": 510,
        "dominant_sentiment": "Positive",
        "dominant_emotion": "joy",
        "engagement_avg": 7.9,
        "topics": {"Technology": 40, "Career": 25, "Academics": 20, "Social": 10, "Faculty": 5, "Infrastructure": 0, "Mental health": 0},
        "transcript": [
            ("Interviewer", "Mark, welcome! Anong nag-inspire sa iyo na mag-apply sa IT?", "Neutral", {"joy": 0.15, "surprise": 0.05, "sadness": 0.02, "anger": 0.01, "fear": 0.01, "love": 0.03}, 7.0),
            ("Student", "Salamat po! Na-hook po ako sa IT nung nag-build ako ng gaming PC ko. Doon ko po na-realize na gusto ko matuto pa ng deeper tungkol sa technology.", "Positive", {"joy": 0.80, "surprise": 0.05, "sadness": 0.01, "anger": 0.01, "fear": 0.01, "love": 0.08}, 8.5),
            ("Interviewer", "So you're into hardware as well?", "Neutral", {"joy": 0.12, "surprise": 0.10, "sadness": 0.02, "anger": 0.01, "fear": 0.01, "love": 0.02}, 7.0),
            ("Student", "Opo! Networking at hardware troubleshooting po ang forte ko. Nag-A+ ako sa isang online training for computer repair.", "Positive", {"joy": 0.72, "surprise": 0.08, "sadness": 0.02, "anger": 0.01, "fear": 0.02, "love": 0.05}, 8.2),
            ("Interviewer", "Maganda yan! What role do you see yourself in after college?", "Positive", {"joy": 0.30, "surprise": 0.05, "sadness": 0.02, "anger": 0.01, "fear": 0.01, "love": 0.05}, 7.5),
            ("Student", "Gusto ko po maging IT consultant o kaya systems administrator ng isang malaking company. Long-term po, dream ko mag-start ng sariling IT solutions business.", "Positive", {"joy": 0.78, "surprise": 0.05, "sadness": 0.01, "anger": 0.01, "fear": 0.03, "love": 0.08}, 8.8),
        ]
    },

    # ── Interview 11: Negative exit, IT ──────────────────────────────
    {
        "date": "2026-03-12 11:15:00",
        "type": "exit",
        "student": "Grace Lim",
        "interviewer": "Dr. Reyes",
        "program": "Information Technology",
        "cohort": "2022",
        "duration": 960,
        "dominant_sentiment": "Negative",
        "dominant_emotion": "sadness",
        "engagement_avg": 5.3,
        "topics": {"Mental health": 30, "Faculty": 25, "Social": 20, "Academics": 15, "Career": 5, "Infrastructure": 5, "Technology": 0},
        "transcript": [
            ("Interviewer", "Grace, let's talk about your experience in the IT program.", "Neutral", {"joy": 0.08, "surprise": 0.05, "sadness": 0.05, "anger": 0.02, "fear": 0.02, "love": 0.01}, 6.0),
            ("Student", "Mahirap po siya honestly. Hindi lang po yung academics, pati yung mental health ko po na-affect.", "Negative", {"joy": 0.02, "surprise": 0.05, "sadness": 0.55, "anger": 0.10, "fear": 0.15, "love": 0.01}, 4.5),
            ("Interviewer", "I'm sorry to hear that. Can you elaborate?", "Neutral", {"joy": 0.05, "surprise": 0.05, "sadness": 0.15, "anger": 0.02, "fear": 0.02, "love": 0.05}, 6.0),
            ("Student", "Sobrang dami po ng workload, lalo na yung thesis semester. Walang po halos tulog. Tapos yung thesis adviser namin, hindi po responsive. Weeks bago mag-reply.", "Negative", {"joy": 0.01, "surprise": 0.05, "sadness": 0.45, "anger": 0.30, "fear": 0.10, "love": 0.01}, 3.8),
            ("Student", "Yung mga kaklase ko rin po, marami ang nagka-anxiety at burnout. Sana po may mental health support na available sa students.", "Negative", {"joy": 0.02, "surprise": 0.05, "sadness": 0.50, "anger": 0.15, "fear": 0.15, "love": 0.05}, 4.0),
            ("Interviewer", "That is important feedback. Anything else?", "Neutral", {"joy": 0.05, "surprise": 0.05, "sadness": 0.08, "anger": 0.02, "fear": 0.02, "love": 0.02}, 5.5),
            ("Student", "Sana po mag-improve yung student-teacher ratio. Sobrang dami namin sa isang section, mahirap po magtanong sa prof kasi ang haba ng pila.", "Negative", {"joy": 0.03, "surprise": 0.05, "sadness": 0.40, "anger": 0.30, "fear": 0.08, "love": 0.01}, 4.2),
            ("Student", "Pero thankful pa rin po ako na natapos ko. Natuto rin po ako maging resilient.", "Neutral", {"joy": 0.25, "surprise": 0.05, "sadness": 0.25, "anger": 0.05, "fear": 0.05, "love": 0.10}, 6.8),
        ]
    },

    # ── Interview 12: Positive admission, CS ─────────────────────────
    {
        "date": "2026-03-13 10:00:00",
        "type": "admission",
        "student": "Kenneth Tan",
        "interviewer": "Prof. Villanueva",
        "program": "Computer Science",
        "cohort": "2026",
        "duration": 570,
        "dominant_sentiment": "Positive",
        "dominant_emotion": "surprise",
        "engagement_avg": 8.1,
        "topics": {"Technology": 35, "Academics": 30, "Career": 20, "Social": 10, "Faculty": 5, "Infrastructure": 0, "Mental health": 0},
        "transcript": [
            ("Interviewer", "Kenneth, bakit ka nag-apply sa CS?", "Neutral", {"joy": 0.10, "surprise": 0.08, "sadness": 0.02, "anger": 0.01, "fear": 0.01, "love": 0.01}, 6.5),
            ("Student", "Na-amaze po kasi ako sa mga AI tools ngayon like ChatGPT. Gusto ko po malaman kung paano sila gumagana behind the scenes.", "Positive", {"joy": 0.60, "surprise": 0.25, "sadness": 0.01, "anger": 0.01, "fear": 0.02, "love": 0.05}, 8.5),
            ("Interviewer", "Interesting perspective! Have you tried building anything AI-related?", "Neutral", {"joy": 0.15, "surprise": 0.10, "sadness": 0.02, "anger": 0.01, "fear": 0.01, "love": 0.02}, 7.0),
            ("Student", "Nag-train po ako ng simple chatbot gamit ang Python at NLTK sa isang online tutorial. Hindi pa po perfect pero nagana naman siya.", "Positive", {"joy": 0.70, "surprise": 0.12, "sadness": 0.02, "anger": 0.01, "fear": 0.03, "love": 0.05}, 8.8),
            ("Student", "Na-surprise po ako kung gaano ka-accessible ang AI tools ngayon. Kahit high school student puwede nang makapag-experiment.", "Positive", {"joy": 0.55, "surprise": 0.30, "sadness": 0.02, "anger": 0.01, "fear": 0.02, "love": 0.05}, 8.2),
            ("Interviewer", "That shows initiative. We look forward to seeing what you build here.", "Positive", {"joy": 0.45, "surprise": 0.10, "sadness": 0.01, "anger": 0.01, "fear": 0.01, "love": 0.08}, 7.5),
        ]
    },

    # ── Interview 13: Neutral exit, CS ───────────────────────────────
    {
        "date": "2026-03-15 14:00:00",
        "type": "exit",
        "student": "Bianca Flores",
        "interviewer": "Dr. Reyes",
        "program": "Computer Science",
        "cohort": "2022",
        "duration": 750,
        "dominant_sentiment": "Neutral",
        "dominant_emotion": "joy",
        "engagement_avg": 7.3,
        "topics": {"Career": 35, "Academics": 25, "Social": 15, "Technology": 10, "Faculty": 10, "Mental health": 5, "Infrastructure": 0},
        "transcript": [
            ("Interviewer", "Bianca, kumusta ang naging experience mo sa CS?", "Neutral", {"joy": 0.10, "surprise": 0.05, "sadness": 0.03, "anger": 0.01, "fear": 0.01, "love": 0.02}, 6.5),
            ("Student", "Maganda naman po overall. May mga hirap pero naging worth it naman po lahat.", "Positive", {"joy": 0.55, "surprise": 0.05, "sadness": 0.08, "anger": 0.02, "fear": 0.02, "love": 0.08}, 7.5),
            ("Interviewer", "What subject prepared you most for your career?", "Neutral", {"joy": 0.08, "surprise": 0.08, "sadness": 0.03, "anger": 0.01, "fear": 0.01, "love": 0.01}, 6.8),
            ("Student", "Software Engineering po at yung Capstone. Doon po namin na-practice yung agile methodology at collaboration skills na kailangan sa industry.", "Positive", {"joy": 0.60, "surprise": 0.10, "sadness": 0.02, "anger": 0.01, "fear": 0.02, "love": 0.05}, 8.0),
            ("Interviewer", "Where will you be working after graduation?", "Neutral", {"joy": 0.10, "surprise": 0.08, "sadness": 0.03, "anger": 0.01, "fear": 0.02, "love": 0.01}, 7.0),
            ("Student", "May offer po ako sa isang startup dito sa Manila. Mag-start po ako bilang backend developer. Medyo kinakabahan pa rin po pero excited.", "Neutral", {"joy": 0.35, "surprise": 0.10, "sadness": 0.05, "anger": 0.02, "fear": 0.15, "love": 0.05}, 7.5),
            ("Student", "Sana lang po mas maraming industry talks ang i-organize ng department para ma-expose ang students sa real-world work environment.", "Neutral", {"joy": 0.15, "surprise": 0.05, "sadness": 0.15, "anger": 0.08, "fear": 0.05, "love": 0.02}, 7.0),
        ]
    },

    # ── Interview 14: Negative admission, IT ─────────────────────────
    {
        "date": "2026-03-17 11:30:00",
        "type": "admission",
        "student": "Daniel Pascual",
        "interviewer": "Prof. Garcia",
        "program": "Information Technology",
        "cohort": "2026",
        "duration": 420,
        "dominant_sentiment": "Negative",
        "dominant_emotion": "sadness",
        "engagement_avg": 4.5,
        "topics": {"Mental health": 40, "Academics": 25, "Social": 15, "Career": 10, "Faculty": 5, "Technology": 5, "Infrastructure": 0},
        "transcript": [
            ("Interviewer", "Daniel, tell us about yourself and why you chose IT.", "Neutral", {"joy": 0.08, "surprise": 0.05, "sadness": 0.05, "anger": 0.02, "fear": 0.03, "love": 0.01}, 6.0),
            ("Student", "To be honest po, ito lang po yung program na may slot pa nung nag-apply ako. First choice ko po sana ay nursing pero puno na.", "Negative", {"joy": 0.03, "surprise": 0.05, "sadness": 0.50, "anger": 0.15, "fear": 0.15, "love": 0.02}, 3.8),
            ("Interviewer", "I see. Do you have any interest in technology at all?", "Neutral", {"joy": 0.08, "surprise": 0.08, "sadness": 0.08, "anger": 0.02, "fear": 0.02, "love": 0.01}, 5.5),
            ("Student", "Ginagamit ko naman po yung computer at phone pero hindi po ako nagco-code o kung ano man. Natatakot po ako baka hindi ko kaya.", "Negative", {"joy": 0.05, "surprise": 0.05, "sadness": 0.35, "anger": 0.05, "fear": 0.40, "love": 0.02}, 3.5),
            ("Interviewer", "We do have support systems in place para sa students na bago pa lang. Hindi ka nag-iisa.", "Positive", {"joy": 0.25, "surprise": 0.05, "sadness": 0.05, "anger": 0.01, "fear": 0.02, "love": 0.10}, 7.0),
            ("Student", "Sige po, try ko po. Salamat po sa encouragement. Sana po kayanin ko.", "Neutral", {"joy": 0.15, "surprise": 0.05, "sadness": 0.30, "anger": 0.03, "fear": 0.25, "love": 0.05}, 5.0),
        ]
    },

    # ── Interview 15: Positive exit, IT ──────────────────────────────
    {
        "date": "2026-03-20 10:00:00",
        "type": "exit",
        "student": "Andrea Navarro",
        "interviewer": "Prof. Villanueva",
        "program": "Information Technology",
        "cohort": "2022",
        "duration": 870,
        "dominant_sentiment": "Positive",
        "dominant_emotion": "love",
        "engagement_avg": 9.0,
        "topics": {"Faculty": 30, "Academics": 25, "Career": 20, "Social": 15, "Technology": 5, "Mental health": 5, "Infrastructure": 0},
        "transcript": [
            ("Interviewer", "Andrea, isa ka sa mga honor students namin. Kumusta ang overall experience mo?", "Positive", {"joy": 0.40, "surprise": 0.05, "sadness": 0.01, "anger": 0.01, "fear": 0.01, "love": 0.10}, 8.0),
            ("Student", "Salamat po! Ang IT program po talaga ang nag-shape sa akin bilang isang professional. Ang dami ko pong natutunan beyond coding.", "Positive", {"joy": 0.82, "surprise": 0.03, "sadness": 0.01, "anger": 0.01, "fear": 0.01, "love": 0.12}, 9.5),
            ("Interviewer", "What was your most memorable experience?", "Neutral", {"joy": 0.15, "surprise": 0.10, "sadness": 0.02, "anger": 0.01, "fear": 0.01, "love": 0.03}, 7.0),
            ("Student", "Yung hackathon po na sinalihan namin kung saan nanalo kami ng first place. Yun po yung moment na na-realize ko na kaya ko pala.", "Positive", {"joy": 0.85, "surprise": 0.05, "sadness": 0.01, "anger": 0.01, "fear": 0.01, "love": 0.07}, 9.8),
            ("Student", "At syempre po yung mga professors namin. Si Ma'am Santos po, parang nanay namin siya sa department. Laging available at supportive.", "Positive", {"joy": 0.70, "surprise": 0.03, "sadness": 0.02, "anger": 0.01, "fear": 0.01, "love": 0.23}, 9.2),
            ("Interviewer", "Beautiful words! Any advice for incoming students?", "Positive", {"joy": 0.35, "surprise": 0.05, "sadness": 0.02, "anger": 0.01, "fear": 0.01, "love": 0.15}, 8.0),
            ("Student", "Wag po kayong matakot magtanong at humingi ng tulong. At i-enjoy po nila ang journey kasi hindi lang po about grades. Yung connections at experiences po ang mahalaga.", "Positive", {"joy": 0.75, "surprise": 0.05, "sadness": 0.02, "anger": 0.01, "fear": 0.02, "love": 0.15}, 9.5),
            ("Student", "Mahal ko po ang school na ito. Forever grateful po ako.", "Positive", {"joy": 0.65, "surprise": 0.02, "sadness": 0.05, "anger": 0.01, "fear": 0.01, "love": 0.26}, 9.8),
        ]
    },
]


# ============================================================================
# SEED SCRIPT
# ============================================================================

def generate_room_id():
    """Generate a unique 8-char room ID."""
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))


def clear_seed_data(cursor):
    """Remove previously seeded data (preserve real user data and real interviews)."""
    # Delete seeded interviews (IDs > 100 are seed data to avoid collision)
    cursor.execute("DELETE FROM topic_classifications WHERE interview_id IN (SELECT id FROM interviews WHERE student_name IN (?))", ("__SEED__",))
    # Simpler approach: delete all dummy data by known student names
    student_names = [iv["student"] for iv in INTERVIEWS]
    placeholders = ",".join("?" * len(student_names))
    
    # Get interview IDs for these students
    cursor.execute(f"SELECT id FROM interviews WHERE student_name IN ({placeholders})", student_names)
    interview_ids = [row[0] for row in cursor.fetchall()]
    
    if interview_ids:
        id_placeholders = ",".join("?" * len(interview_ids))
        cursor.execute(f"DELETE FROM topic_classifications WHERE interview_id IN ({id_placeholders})", interview_ids)
        cursor.execute(f"DELETE FROM interview_summary WHERE interview_id IN ({id_placeholders})", interview_ids)
        cursor.execute(f"DELETE FROM analysis_results WHERE interview_id IN ({id_placeholders})", interview_ids)
        cursor.execute(f"DELETE FROM transcripts WHERE interview_id IN ({id_placeholders})", interview_ids)
        cursor.execute(f"DELETE FROM interviews WHERE id IN ({id_placeholders})", interview_ids)
        print(f"   Cleared {len(interview_ids)} previously seeded interviews")


def seed_database():
    """Main seeding function."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    print("\n" + "=" * 60)
    print("   DATABASE SEED SCRIPT")
    print("=" * 60)

    # Clear old seed data
    clear_seed_data(cursor)
    conn.commit()

    total_transcripts = 0
    total_analysis = 0

    for i, iv in enumerate(INTERVIEWS, 1):
        room_id = generate_room_id()
        created_at = iv["date"]
        started_at = iv["date"]

        # Calculate ended_at
        start_dt = datetime.strptime(iv["date"], "%Y-%m-%d %H:%M:%S")
        end_dt = start_dt + timedelta(seconds=iv["duration"])
        ended_at = end_dt.strftime("%Y-%m-%d %H:%M:%S")

        # 1. Insert interview
        cursor.execute('''
            INSERT INTO interviews (room_id, interview_type, student_name, interviewer_name,
                                    program, cohort, status, created_at, started_at, ended_at, duration_seconds)
            VALUES (?, ?, ?, ?, ?, ?, 'completed', ?, ?, ?, ?)
        ''', (room_id, iv["type"], iv["student"], iv["interviewer"],
              iv["program"], iv["cohort"], created_at, started_at, ended_at, iv["duration"]))

        interview_id = cursor.lastrowid

        # 2. Insert transcripts + analysis per line
        line_count = len(iv["transcript"])
        for j, (speaker, text, sent_label, emotions, engagement) in enumerate(iv["transcript"]):
            # Transcript timestamp spaced evenly across the interview duration
            offset = timedelta(seconds=int(iv["duration"] * (j / max(line_count - 1, 1))))
            line_ts = (start_dt + offset).strftime("%Y-%m-%d %H:%M:%S")

            cursor.execute('''
                INSERT INTO transcripts (interview_id, speaker, text, timestamp)
                VALUES (?, ?, ?, ?)
            ''', (interview_id, speaker, text, line_ts))
            transcript_id = cursor.lastrowid
            total_transcripts += 1

            # Sentiment probabilities based on label
            if sent_label == "Positive":
                pos = round(random.uniform(0.55, 0.92), 2)
                neg = round(random.uniform(0.01, 0.10), 2)
                neu = round(1.0 - pos - neg, 2)
            elif sent_label == "Negative":
                neg = round(random.uniform(0.50, 0.88), 2)
                pos = round(random.uniform(0.01, 0.10), 2)
                neu = round(1.0 - neg - pos, 2)
            else:
                neu = round(random.uniform(0.45, 0.70), 2)
                pos = round(random.uniform(0.10, 0.30), 2)
                neg = round(1.0 - neu - pos, 2)

            confidence = max(pos, neu, neg)

            # Engagement level
            if engagement >= 8.0:
                eng_level = "High"
            elif engagement >= 5.5:
                eng_level = "Medium"
            else:
                eng_level = "Low"

            # Extract top keyphrases from text
            words = text.split()
            keyphrases = []
            if len(words) > 3:
                # Pick 2-3 random bigrams as keyphrases
                for _ in range(min(3, len(words) - 1)):
                    idx = random.randint(0, len(words) - 2)
                    keyphrases.append(f"{words[idx]} {words[idx+1]}")

            cursor.execute('''
                INSERT INTO analysis_results (
                    interview_id, transcript_id,
                    sentiment_label, sentiment_confidence,
                    sentiment_positive, sentiment_neutral, sentiment_negative,
                    emotions_json, keyphrases_json,
                    engagement_score, engagement_level,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                interview_id, transcript_id,
                sent_label, confidence,
                pos, neu, neg,
                json.dumps(emotions),
                json.dumps(keyphrases),
                engagement, eng_level,
                line_ts
            ))
            total_analysis += 1

        # 3. Insert interview summary
        cursor.execute('''
            INSERT INTO interview_summary (
                interview_id, total_words, avg_sentiment_score,
                dominant_sentiment, dominant_emotion,
                top_topics_json, avg_engagement_score, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            interview_id,
            sum(len(t[1].split()) for t in iv["transcript"]),
            line_count,  # count of analysis rows
            iv["dominant_sentiment"],
            iv["dominant_emotion"],
            json.dumps(iv["topics"]),
            iv["engagement_avg"],
            created_at
        ))

        # 4. Insert topic classification
        topics = iv["topics"]
        total_sentences = line_count
        classified_sentences = line_count
        cursor.execute('''
            INSERT INTO topic_classifications (
                interview_id, academics_percent, career_percent, faculty_percent,
                infrastructure_percent, mental_health_percent, social_percent,
                technology_percent, total_sentences, classified_sentences, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            interview_id,
            topics.get("Academics", 0),
            topics.get("Career", 0),
            topics.get("Faculty", 0),
            topics.get("Infrastructure", 0),
            topics.get("Mental health", 0),
            topics.get("Social", 0),
            topics.get("Technology", 0),
            total_sentences,
            classified_sentences,
            created_at
        ))

        print(f"   [{i:2d}/15] {iv['student']:<25s} | {iv['type']:10s} | {iv['dominant_sentiment']:8s} | {iv['date'][:10]}")

    conn.commit()
    conn.close()

    print("-" * 60)
    print(f"   Interviews:    15")
    print(f"   Transcripts:   {total_transcripts}")
    print(f"   Analysis rows: {total_analysis}")
    print(f"   Summaries:     15")
    print(f"   Topic classes: 15")
    print("=" * 60)
    print("   ✅ Database seeded successfully!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    seed_database()
