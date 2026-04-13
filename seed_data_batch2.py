"""
Seed Script Batch 2 - Add more interviews on existing dates
============================================================
Adds 15 MORE interviews on the SAME dates as Batch 1 so the
stacked bar chart shows multiple interviews per date.
Does NOT delete existing data.
"""

import sqlite3, json, random, string
from datetime import datetime, timedelta

DB_PATH = "interviews.db"

INTERVIEWS_BATCH2 = [
    # ── Feb 15: 2nd interview (Negative admission) ──────────────
    {
        "date": "2026-02-15 11:00:00",
        "type": "admission",
        "student": "Rico Magbanua",
        "interviewer": "Dr. Reyes",
        "program": "Information Technology",
        "cohort": "2026",
        "duration": 480,
        "dominant_sentiment": "Negative",
        "dominant_emotion": "fear",
        "engagement_avg": 4.6,
        "topics": {"Mental health": 35, "Academics": 30, "Career": 15, "Social": 10, "Technology": 5, "Faculty": 5, "Infrastructure": 0},
        "transcript": [
            ("Interviewer", "Rico, what brings you to the IT program?", "Neutral", {"joy": 0.10, "surprise": 0.05, "sadness": 0.05, "anger": 0.02, "fear": 0.05, "love": 0.01}, 5.5),
            ("Student", "Wala po akong ibang choice. Sinabi ng magulang ko na dito na lang ako kasi malapit sa bahay namin.", "Negative", {"joy": 0.03, "surprise": 0.05, "sadness": 0.45, "anger": 0.15, "fear": 0.20, "love": 0.02}, 3.8),
            ("Interviewer", "Do you have any experience with computers?", "Neutral", {"joy": 0.08, "surprise": 0.05, "sadness": 0.03, "anger": 0.01, "fear": 0.02, "love": 0.01}, 5.8),
            ("Student", "Social media lang po. Hindi po ako marunong mag-code. Sobrang natatakot po ako na baka ma-fail ako.", "Negative", {"joy": 0.02, "surprise": 0.05, "sadness": 0.30, "anger": 0.08, "fear": 0.48, "love": 0.01}, 3.5),
            ("Student", "Sana po may remedial classes para sa mga katulad ko na walang background sa computers.", "Neutral", {"joy": 0.10, "surprise": 0.05, "sadness": 0.25, "anger": 0.05, "fear": 0.30, "love": 0.02}, 5.0),
        ]
    },
    # ── Feb 15: 3rd interview (Neutral admission) ───────────────
    {
        "date": "2026-02-15 14:30:00",
        "type": "admission",
        "student": "Janelle Ocampo",
        "interviewer": "Prof. Villanueva",
        "program": "Computer Science",
        "cohort": "2026",
        "duration": 550,
        "dominant_sentiment": "Neutral",
        "dominant_emotion": "surprise",
        "engagement_avg": 6.5,
        "topics": {"Academics": 40, "Technology": 25, "Career": 20, "Social": 10, "Faculty": 5, "Infrastructure": 0, "Mental health": 0},
        "transcript": [
            ("Interviewer", "Janelle, kamusta! Why Computer Science?", "Neutral", {"joy": 0.12, "surprise": 0.08, "sadness": 0.03, "anger": 0.01, "fear": 0.01, "love": 0.02}, 6.0),
            ("Student", "Curious lang po ako sa programming. Nag-try po ako ng short course online, okay naman po siya.", "Neutral", {"joy": 0.25, "surprise": 0.20, "sadness": 0.05, "anger": 0.02, "fear": 0.08, "love": 0.03}, 6.5),
            ("Interviewer", "What did you learn in that course?", "Neutral", {"joy": 0.10, "surprise": 0.08, "sadness": 0.02, "anger": 0.01, "fear": 0.01, "love": 0.01}, 6.2),
            ("Student", "Basic HTML at CSS po. Na-surprise po ako na kaya ko pala gumawa ng simple webpage. Medyo interesting po siya.", "Positive", {"joy": 0.35, "surprise": 0.35, "sadness": 0.02, "anger": 0.01, "fear": 0.03, "love": 0.05}, 7.5),
            ("Student", "Hindi ko pa po sure kung ito talaga gusto ko pero willing naman po ako matuto.", "Neutral", {"joy": 0.15, "surprise": 0.10, "sadness": 0.10, "anger": 0.02, "fear": 0.15, "love": 0.02}, 6.0),
        ]
    },
    # ── Feb 22: 2nd interview (Negative exit) ───────────────────
    {
        "date": "2026-02-22 14:00:00",
        "type": "exit",
        "student": "Miguel Ramos",
        "interviewer": "Dr. Reyes",
        "program": "Information Technology",
        "cohort": "2022",
        "duration": 900,
        "dominant_sentiment": "Negative",
        "dominant_emotion": "anger",
        "engagement_avg": 4.9,
        "topics": {"Infrastructure": 40, "Faculty": 25, "Academics": 15, "Mental health": 10, "Career": 5, "Social": 5, "Technology": 0},
        "transcript": [
            ("Interviewer", "Miguel, how was your IT experience?", "Neutral", {"joy": 0.08, "surprise": 0.05, "sadness": 0.05, "anger": 0.03, "fear": 0.02, "love": 0.01}, 5.5),
            ("Student", "Masasabi ko po na kulang yung resources ng school. Luma na po yung mga gamit sa lab, hindi na po sapat.", "Negative", {"joy": 0.02, "surprise": 0.05, "sadness": 0.30, "anger": 0.45, "fear": 0.05, "love": 0.01}, 4.0),
            ("Student", "Tapos yung WiFi po sa campus, halos walang kwenta. Paano kami mag-research kung ganoon?", "Negative", {"joy": 0.01, "surprise": 0.05, "sadness": 0.25, "anger": 0.55, "fear": 0.03, "love": 0.01}, 3.5),
            ("Interviewer", "Any suggestions for improvement?", "Neutral", {"joy": 0.08, "surprise": 0.05, "sadness": 0.05, "anger": 0.02, "fear": 0.02, "love": 0.01}, 5.8),
            ("Student", "Mag-invest po sana sila sa bagong equipment. Yung mga computers po namin, 2017 pa po yung specs. Hindi na po kaya ng modern software.", "Negative", {"joy": 0.02, "surprise": 0.08, "sadness": 0.30, "anger": 0.45, "fear": 0.05, "love": 0.01}, 4.2),
            ("Student", "At sana po ayusin ang scheduling, sobrang cramped ng mga klase namin.", "Negative", {"joy": 0.03, "surprise": 0.05, "sadness": 0.35, "anger": 0.40, "fear": 0.05, "love": 0.01}, 4.5),
        ]
    },
    # ── Feb 25: 2nd interview (Positive admission) ──────────────
    {
        "date": "2026-02-25 14:00:00",
        "type": "admission",
        "student": "Christine Lagman",
        "interviewer": "Prof. Villanueva",
        "program": "Computer Science",
        "cohort": "2026",
        "duration": 600,
        "dominant_sentiment": "Positive",
        "dominant_emotion": "joy",
        "engagement_avg": 8.5,
        "topics": {"Technology": 35, "Career": 30, "Academics": 20, "Social": 10, "Faculty": 5, "Infrastructure": 0, "Mental health": 0},
        "transcript": [
            ("Interviewer", "Christine, tell us about your interest in CS.", "Neutral", {"joy": 0.12, "surprise": 0.05, "sadness": 0.02, "anger": 0.01, "fear": 0.01, "love": 0.02}, 6.5),
            ("Student", "Super excited po ako! Nag-develop na po ako ng mobile app para sa barangay namin, records tracking po siya.", "Positive", {"joy": 0.88, "surprise": 0.03, "sadness": 0.01, "anger": 0.01, "fear": 0.01, "love": 0.06}, 9.2),
            ("Student", "Gamit ko po Flutter at Dart. Self-taught po ako pero gusto ko na mag-formalize ng knowledge ko.", "Positive", {"joy": 0.75, "surprise": 0.08, "sadness": 0.01, "anger": 0.01, "fear": 0.03, "love": 0.05}, 8.8),
            ("Interviewer", "That's impressive! What are your career goals?", "Positive", {"joy": 0.40, "surprise": 0.15, "sadness": 0.01, "anger": 0.01, "fear": 0.01, "love": 0.05}, 7.5),
            ("Student", "Gusto ko po mag-work sa Google o sa local tech companies na gumagawa ng products para sa Pilipinas.", "Positive", {"joy": 0.80, "surprise": 0.05, "sadness": 0.01, "anger": 0.01, "fear": 0.02, "love": 0.08}, 9.0),
        ]
    },
    # ── Mar 2: 2nd interview (Neutral exit) ─────────────────────
    {
        "date": "2026-03-02 14:00:00",
        "type": "exit",
        "student": "Jerome Santos",
        "interviewer": "Prof. Garcia",
        "program": "Computer Science",
        "cohort": "2022",
        "duration": 720,
        "dominant_sentiment": "Neutral",
        "dominant_emotion": "sadness",
        "engagement_avg": 6.2,
        "topics": {"Academics": 30, "Career": 25, "Faculty": 20, "Social": 10, "Technology": 10, "Mental health": 5, "Infrastructure": 0},
        "transcript": [
            ("Interviewer", "Jerome, kumusta ang naging four years mo?", "Neutral", {"joy": 0.10, "surprise": 0.05, "sadness": 0.05, "anger": 0.02, "fear": 0.01, "love": 0.02}, 6.0),
            ("Student", "May mga maganda at may mga hindi po. Yung first two years okay naman, pero yung last two medyo struggle.", "Neutral", {"joy": 0.15, "surprise": 0.08, "sadness": 0.30, "anger": 0.08, "fear": 0.05, "love": 0.03}, 5.5),
            ("Interviewer", "What was the biggest challenge?", "Neutral", {"joy": 0.05, "surprise": 0.08, "sadness": 0.08, "anger": 0.02, "fear": 0.02, "love": 0.01}, 6.0),
            ("Student", "Yung thesis po. Isang taon po halos bago namin natapos. Ang hirap po mag-coordinate ng schedule ng lahat ng members.", "Negative", {"joy": 0.03, "surprise": 0.05, "sadness": 0.45, "anger": 0.20, "fear": 0.10, "love": 0.02}, 5.0),
            ("Student", "Pero natuto naman po ako maraming bagay. Mas confident na po ako ngayon sa programming.", "Positive", {"joy": 0.50, "surprise": 0.05, "sadness": 0.08, "anger": 0.02, "fear": 0.02, "love": 0.08}, 7.8),
        ]
    },
    # ── Mar 4: 2nd interview (Negative admission) ──────────────
    {
        "date": "2026-03-04 10:00:00",
        "type": "admission",
        "student": "Alyssa Mendez",
        "interviewer": "Dr. Reyes",
        "program": "Information Technology",
        "cohort": "2026",
        "duration": 450,
        "dominant_sentiment": "Negative",
        "dominant_emotion": "sadness",
        "engagement_avg": 4.3,
        "topics": {"Mental health": 40, "Social": 25, "Academics": 20, "Career": 10, "Faculty": 5, "Technology": 0, "Infrastructure": 0},
        "transcript": [
            ("Interviewer", "Alyssa, why did you choose IT?", "Neutral", {"joy": 0.08, "surprise": 0.05, "sadness": 0.05, "anger": 0.02, "fear": 0.03, "love": 0.01}, 5.5),
            ("Student", "Hindi ko po talaga alam kung anong gusto ko. Sinunod ko na lang po yung suggestion ng guidance counselor namin.", "Negative", {"joy": 0.03, "surprise": 0.05, "sadness": 0.50, "anger": 0.10, "fear": 0.20, "love": 0.02}, 3.5),
            ("Student", "Nahihirapan po kasi ako socially. Sana lang po may supportive community dito.", "Negative", {"joy": 0.05, "surprise": 0.05, "sadness": 0.45, "anger": 0.05, "fear": 0.25, "love": 0.05}, 3.8),
            ("Interviewer", "We have student organizations that can help you connect with peers.", "Positive", {"joy": 0.25, "surprise": 0.05, "sadness": 0.05, "anger": 0.01, "fear": 0.02, "love": 0.10}, 7.0),
            ("Student", "Sige po, try ko nga po. Salamat po.", "Neutral", {"joy": 0.15, "surprise": 0.05, "sadness": 0.25, "anger": 0.03, "fear": 0.18, "love": 0.05}, 5.0),
        ]
    },
    # ── Mar 8: 2nd interview (Positive admission) ──────────────
    {
        "date": "2026-03-08 10:00:00",
        "type": "admission",
        "student": "Rafael Torres",
        "interviewer": "Prof. Garcia",
        "program": "Computer Science",
        "cohort": "2026",
        "duration": 540,
        "dominant_sentiment": "Positive",
        "dominant_emotion": "joy",
        "engagement_avg": 8.3,
        "topics": {"Technology": 40, "Academics": 25, "Career": 25, "Social": 5, "Faculty": 5, "Infrastructure": 0, "Mental health": 0},
        "transcript": [
            ("Interviewer", "Rafael, why CS?", "Neutral", {"joy": 0.10, "surprise": 0.05, "sadness": 0.02, "anger": 0.01, "fear": 0.01, "love": 0.02}, 6.5),
            ("Student", "Gumawa po ako ng robot nung Grade 11 gamit ang Arduino. Doon po nagsimula yung passion ko sa tech.", "Positive", {"joy": 0.85, "surprise": 0.05, "sadness": 0.01, "anger": 0.01, "fear": 0.01, "love": 0.07}, 9.0),
            ("Student", "Nanalo po kami ng regional science fair. Yun po yung turning point sa decision ko.", "Positive", {"joy": 0.80, "surprise": 0.08, "sadness": 0.01, "anger": 0.01, "fear": 0.01, "love": 0.05}, 8.8),
            ("Interviewer", "Excellent! What do you want to focus on?", "Positive", {"joy": 0.35, "surprise": 0.10, "sadness": 0.01, "anger": 0.01, "fear": 0.01, "love": 0.05}, 7.5),
            ("Student", "Robotics at IoT po. Gusto ko po gumawa ng smart farming solutions para sa mga magsasaka sa probinsya namin.", "Positive", {"joy": 0.78, "surprise": 0.05, "sadness": 0.01, "anger": 0.01, "fear": 0.02, "love": 0.13}, 9.2),
        ]
    },
    # ── Mar 10: 2nd interview (Neutral exit) ────────────────────
    {
        "date": "2026-03-10 14:00:00",
        "type": "exit",
        "student": "Hannah Velasco",
        "interviewer": "Prof. Villanueva",
        "program": "Information Technology",
        "cohort": "2022",
        "duration": 810,
        "dominant_sentiment": "Neutral",
        "dominant_emotion": "surprise",
        "engagement_avg": 6.8,
        "topics": {"Social": 30, "Career": 25, "Academics": 20, "Faculty": 10, "Technology": 10, "Mental health": 5, "Infrastructure": 0},
        "transcript": [
            ("Interviewer", "Hannah, how was your time in IT?", "Neutral", {"joy": 0.10, "surprise": 0.08, "sadness": 0.03, "anger": 0.01, "fear": 0.01, "love": 0.02}, 6.5),
            ("Student", "Mixed feelings po. Maganda yung friendships na nabuo pero yung curriculum medyo hindi updated.", "Neutral", {"joy": 0.20, "surprise": 0.15, "sadness": 0.15, "anger": 0.08, "fear": 0.02, "love": 0.10}, 6.5),
            ("Student", "Na-surprise po ako na yung pinakamatutun ko pala ay galing sa mga extracurricular activities, hindi sa klase.", "Neutral", {"joy": 0.25, "surprise": 0.35, "sadness": 0.08, "anger": 0.05, "fear": 0.02, "love": 0.05}, 7.0),
            ("Interviewer", "What activities were most valuable?", "Neutral", {"joy": 0.10, "surprise": 0.10, "sadness": 0.03, "anger": 0.01, "fear": 0.01, "love": 0.01}, 6.5),
            ("Student", "Yung Google Developer Student Club po. Doon ko natutunan ang modern web frameworks na hindi tinuturo sa klase.", "Positive", {"joy": 0.45, "surprise": 0.20, "sadness": 0.05, "anger": 0.02, "fear": 0.02, "love": 0.08}, 7.8),
            ("Student", "Sana po i-integrate na nila yun sa curriculum para lahat matuto, hindi lang yung mga sumali sa org.", "Neutral", {"joy": 0.10, "surprise": 0.10, "sadness": 0.20, "anger": 0.12, "fear": 0.05, "love": 0.02}, 6.5),
        ]
    },
    # ── Mar 13: 2nd interview (Positive exit) ───────────────────
    {
        "date": "2026-03-13 14:30:00",
        "type": "exit",
        "student": "Camille Dizon",
        "interviewer": "Dr. Reyes",
        "program": "Computer Science",
        "cohort": "2022",
        "duration": 780,
        "dominant_sentiment": "Positive",
        "dominant_emotion": "love",
        "engagement_avg": 8.7,
        "topics": {"Faculty": 35, "Academics": 25, "Social": 20, "Career": 10, "Technology": 5, "Mental health": 5, "Infrastructure": 0},
        "transcript": [
            ("Interviewer", "Camille, congratulations! Kumusta ang CS journey mo?", "Positive", {"joy": 0.40, "surprise": 0.05, "sadness": 0.01, "anger": 0.01, "fear": 0.01, "love": 0.10}, 8.0),
            ("Student", "Sobrang thankful po ako sa mga mentors ko. Si Sir Alvarez po talaga ang nagbago ng perspective ko sa coding.", "Positive", {"joy": 0.72, "surprise": 0.03, "sadness": 0.02, "anger": 0.01, "fear": 0.01, "love": 0.21}, 9.2),
            ("Student", "Dahil sa kanya, nag-shift po ako from backend to full-stack. Ang dami ko pong natutunan na wala sa textbook.", "Positive", {"joy": 0.68, "surprise": 0.10, "sadness": 0.01, "anger": 0.01, "fear": 0.02, "love": 0.15}, 9.0),
            ("Interviewer", "What advice would you give to current students?", "Neutral", {"joy": 0.12, "surprise": 0.05, "sadness": 0.03, "anger": 0.01, "fear": 0.01, "love": 0.03}, 7.0),
            ("Student", "Build projects outside of class! Yun po ang magse-set apart sa inyo. At huwag matakot humingi ng tulong.", "Positive", {"joy": 0.65, "surprise": 0.05, "sadness": 0.02, "anger": 0.01, "fear": 0.03, "love": 0.15}, 8.8),
            ("Student", "Mahal ko po ang CS department. Parang family po kayo sa akin.", "Positive", {"joy": 0.55, "surprise": 0.03, "sadness": 0.05, "anger": 0.01, "fear": 0.01, "love": 0.35}, 9.5),
        ]
    },
    # ── Mar 15: 2nd interview (Negative exit) ───────────────────
    {
        "date": "2026-03-15 10:00:00",
        "type": "exit",
        "student": "Paolo Reyes",
        "interviewer": "Prof. Garcia",
        "program": "Information Technology",
        "cohort": "2022",
        "duration": 660,
        "dominant_sentiment": "Negative",
        "dominant_emotion": "anger",
        "engagement_avg": 5.0,
        "topics": {"Academics": 35, "Infrastructure": 30, "Faculty": 20, "Career": 10, "Mental health": 5, "Social": 0, "Technology": 0},
        "transcript": [
            ("Interviewer", "Paolo, tell us about your experience.", "Neutral", {"joy": 0.08, "surprise": 0.05, "sadness": 0.05, "anger": 0.03, "fear": 0.02, "love": 0.01}, 5.8),
            ("Student", "Prangkahan po tayo. Yung curriculum, parang stuck sa 2018. Wala pong cloud computing, DevOps, o containerization na tinuturo.", "Negative", {"joy": 0.02, "surprise": 0.08, "sadness": 0.25, "anger": 0.50, "fear": 0.03, "love": 0.01}, 4.0),
            ("Student", "Nag-aral po ako mag-Docker at Kubernetes sa sarili ko kasi hindi siya covered. Sayang yung tuition kung ganoon.", "Negative", {"joy": 0.05, "surprise": 0.08, "sadness": 0.30, "anger": 0.42, "fear": 0.05, "love": 0.01}, 4.5),
            ("Interviewer", "What would you change about the program?", "Neutral", {"joy": 0.05, "surprise": 0.05, "sadness": 0.08, "anger": 0.03, "fear": 0.02, "love": 0.01}, 6.0),
            ("Student", "Kumuha po sila ng industry practitioners na mag-teach, hindi lang purely academic. At mag-update ng syllabus every two years.", "Negative", {"joy": 0.08, "surprise": 0.05, "sadness": 0.25, "anger": 0.40, "fear": 0.05, "love": 0.02}, 5.2),
        ]
    },
    # ── Mar 17: 2nd interview (Positive admission) ──────────────
    {
        "date": "2026-03-17 14:00:00",
        "type": "admission",
        "student": "Trisha Mae Gonzales",
        "interviewer": "Prof. Villanueva",
        "program": "Computer Science",
        "cohort": "2026",
        "duration": 510,
        "dominant_sentiment": "Positive",
        "dominant_emotion": "joy",
        "engagement_avg": 8.0,
        "topics": {"Academics": 35, "Technology": 30, "Career": 20, "Social": 10, "Faculty": 5, "Infrastructure": 0, "Mental health": 0},
        "transcript": [
            ("Interviewer", "Trisha Mae, why Computer Science?", "Neutral", {"joy": 0.12, "surprise": 0.05, "sadness": 0.02, "anger": 0.01, "fear": 0.01, "love": 0.02}, 6.5),
            ("Student", "Nag-participate po ako sa Code.org nung elementary pa lang. Since then, never nawala ang interest ko sa coding.", "Positive", {"joy": 0.82, "surprise": 0.05, "sadness": 0.01, "anger": 0.01, "fear": 0.01, "love": 0.08}, 8.5),
            ("Student", "Gumawa na po ako ng portfolio website at ilang Python automation scripts para sa school namin.", "Positive", {"joy": 0.70, "surprise": 0.08, "sadness": 0.02, "anger": 0.01, "fear": 0.03, "love": 0.05}, 8.5),
            ("Interviewer", "Very impressive! What do you hope to learn here?", "Positive", {"joy": 0.40, "surprise": 0.10, "sadness": 0.01, "anger": 0.01, "fear": 0.01, "love": 0.05}, 7.5),
            ("Student", "Data science at cybersecurity po. Gusto ko po maging ethical hacker someday!", "Positive", {"joy": 0.78, "surprise": 0.08, "sadness": 0.01, "anger": 0.01, "fear": 0.02, "love": 0.05}, 8.8),
        ]
    },
    # ── Mar 20: 2nd interview (Neutral exit) ────────────────────
    {
        "date": "2026-03-20 14:00:00",
        "type": "exit",
        "student": "Justin Bernardo",
        "interviewer": "Dr. Reyes",
        "program": "Computer Science",
        "cohort": "2022",
        "duration": 690,
        "dominant_sentiment": "Neutral",
        "dominant_emotion": "surprise",
        "engagement_avg": 7.0,
        "topics": {"Career": 35, "Academics": 25, "Technology": 15, "Social": 10, "Faculty": 10, "Mental health": 5, "Infrastructure": 0},
        "transcript": [
            ("Interviewer", "Justin, how do you feel about graduating?", "Neutral", {"joy": 0.15, "surprise": 0.10, "sadness": 0.05, "anger": 0.01, "fear": 0.02, "love": 0.02}, 6.5),
            ("Student", "Medyo surreal po. Hindi ko akalain na matapos ko rin. May mga times na gusto ko na mag-shift ng course.", "Neutral", {"joy": 0.20, "surprise": 0.30, "sadness": 0.15, "anger": 0.05, "fear": 0.05, "love": 0.05}, 7.0),
            ("Student", "Pero glad po ako na nagstick ako. Yung capstone project namin, na-feature pa po sa school website.", "Positive", {"joy": 0.55, "surprise": 0.15, "sadness": 0.02, "anger": 0.01, "fear": 0.02, "love": 0.08}, 8.0),
            ("Interviewer", "What's next for you?", "Neutral", {"joy": 0.10, "surprise": 0.08, "sadness": 0.03, "anger": 0.01, "fear": 0.02, "love": 0.01}, 6.5),
            ("Student", "May interview na po ako sa isang BPO company for a developer role. Kinakabahan pero excited.", "Neutral", {"joy": 0.30, "surprise": 0.10, "sadness": 0.05, "anger": 0.02, "fear": 0.20, "love": 0.03}, 6.8),
        ]
    },
    # ── Mar 20: 3rd interview (Negative admission) ──────────────
    {
        "date": "2026-03-20 11:00:00",
        "type": "admission",
        "student": "Ella Mae Castillo",
        "interviewer": "Prof. Garcia",
        "program": "Information Technology",
        "cohort": "2026",
        "duration": 420,
        "dominant_sentiment": "Negative",
        "dominant_emotion": "fear",
        "engagement_avg": 4.2,
        "topics": {"Mental health": 45, "Academics": 25, "Social": 15, "Career": 10, "Faculty": 5, "Technology": 0, "Infrastructure": 0},
        "transcript": [
            ("Interviewer", "Ella Mae, bakit ka nag-apply sa IT?", "Neutral", {"joy": 0.08, "surprise": 0.05, "sadness": 0.05, "anger": 0.02, "fear": 0.03, "love": 0.01}, 5.5),
            ("Student", "Nashare po ng friend ko na may scholarship dito. Kaya lang po wala akong laptop, phone lang po gamit ko.", "Negative", {"joy": 0.05, "surprise": 0.05, "sadness": 0.45, "anger": 0.08, "fear": 0.25, "love": 0.02}, 3.5),
            ("Student", "Natatakot po ako na baka masyadong mahal yung mga kailangan for IT. Hindi po kasi kami mayaman.", "Negative", {"joy": 0.02, "surprise": 0.05, "sadness": 0.38, "anger": 0.05, "fear": 0.42, "love": 0.02}, 3.2),
            ("Interviewer", "We have computer labs available for students and financial assistance programs.", "Positive", {"joy": 0.25, "surprise": 0.05, "sadness": 0.05, "anger": 0.01, "fear": 0.02, "love": 0.10}, 7.0),
            ("Student", "Talaga po? Salamat po. Sana po kaya ko, gusto ko po talagang makapagtapos.", "Neutral", {"joy": 0.20, "surprise": 0.10, "sadness": 0.20, "anger": 0.02, "fear": 0.20, "love": 0.08}, 5.0),
        ]
    },
    # ── Mar 6: 2nd interview (Positive exit) ────────────────────
    {
        "date": "2026-03-06 14:30:00",
        "type": "exit",
        "student": "Lovely Navidad",
        "interviewer": "Prof. Villanueva",
        "program": "Computer Science",
        "cohort": "2022",
        "duration": 750,
        "dominant_sentiment": "Positive",
        "dominant_emotion": "joy",
        "engagement_avg": 8.4,
        "topics": {"Career": 30, "Academics": 25, "Technology": 20, "Faculty": 15, "Social": 5, "Mental health": 5, "Infrastructure": 0},
        "transcript": [
            ("Interviewer", "Lovely, kamusta ang experience mo sa CS?", "Neutral", {"joy": 0.15, "surprise": 0.05, "sadness": 0.02, "anger": 0.01, "fear": 0.01, "love": 0.03}, 7.0),
            ("Student", "Best decision ko po na pumili ng CS. Sobrang dami ko pong natutunan na applicable sa real world.", "Positive", {"joy": 0.85, "surprise": 0.03, "sadness": 0.01, "anger": 0.01, "fear": 0.01, "love": 0.09}, 9.0),
            ("Student", "Na-hire na po ako ng Shopee bilang software engineer kahit hindi pa po ako officially graduated!", "Positive", {"joy": 0.88, "surprise": 0.05, "sadness": 0.01, "anger": 0.01, "fear": 0.01, "love": 0.04}, 9.5),
            ("Interviewer", "Wow! That's amazing. What helped you the most?", "Positive", {"joy": 0.45, "surprise": 0.20, "sadness": 0.01, "anger": 0.01, "fear": 0.01, "love": 0.05}, 8.0),
            ("Student", "Yung mga hackathons at coding competitions po. At yung mga professors na nag-encourage sa amin na mag-build ng portfolio.", "Positive", {"joy": 0.75, "surprise": 0.05, "sadness": 0.02, "anger": 0.01, "fear": 0.02, "love": 0.12}, 8.5),
        ]
    },
    # ── Mar 12: 2nd interview (Positive admission) ──────────────
    {
        "date": "2026-03-12 14:30:00",
        "type": "admission",
        "student": "Marc Andrei Cruz",
        "interviewer": "Dr. Reyes",
        "program": "Computer Science",
        "cohort": "2026",
        "duration": 570,
        "dominant_sentiment": "Positive",
        "dominant_emotion": "love",
        "engagement_avg": 8.6,
        "topics": {"Academics": 30, "Career": 30, "Technology": 20, "Social": 10, "Faculty": 5, "Mental health": 5, "Infrastructure": 0},
        "transcript": [
            ("Interviewer", "Marc Andrei, why did you choose CS here?", "Neutral", {"joy": 0.12, "surprise": 0.05, "sadness": 0.02, "anger": 0.01, "fear": 0.01, "love": 0.03}, 6.5),
            ("Student", "Kuya ko po graduate dito sa CS, siya po ang inspiration ko. Nakita ko po kung paano nagbago buhay niya dahil sa degree na ito.", "Positive", {"joy": 0.70, "surprise": 0.05, "sadness": 0.02, "anger": 0.01, "fear": 0.01, "love": 0.21}, 9.0),
            ("Student", "Nagtuturo na po siya sa akin ng Python simula nung Grade 9 pa. Ready na po ako!", "Positive", {"joy": 0.82, "surprise": 0.05, "sadness": 0.01, "anger": 0.01, "fear": 0.02, "love": 0.09}, 9.2),
            ("Interviewer", "What a great story. What do you want to do after graduating?", "Positive", {"joy": 0.30, "surprise": 0.08, "sadness": 0.02, "anger": 0.01, "fear": 0.01, "love": 0.08}, 7.5),
            ("Student", "Mag-startup po kami ng kuya ko. Gusto namin gumawa ng EdTech platform para sa public schools.", "Positive", {"joy": 0.78, "surprise": 0.05, "sadness": 0.01, "anger": 0.01, "fear": 0.02, "love": 0.13}, 9.0),
        ]
    },
]


def generate_room_id():
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))


def seed_batch2():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    print("\n" + "=" * 60)
    print("   SEED BATCH 2 - Adding 15 more interviews")
    print("=" * 60)

    total_transcripts = 0
    total_analysis = 0

    for i, iv in enumerate(INTERVIEWS_BATCH2, 1):
        room_id = generate_room_id()
        start_dt = datetime.strptime(iv["date"], "%Y-%m-%d %H:%M:%S")
        end_dt = start_dt + timedelta(seconds=iv["duration"])

        cursor.execute('''
            INSERT INTO interviews (room_id, interview_type, student_name, interviewer_name,
                                    program, cohort, status, created_at, started_at, ended_at, duration_seconds)
            VALUES (?, ?, ?, ?, ?, ?, 'completed', ?, ?, ?, ?)
        ''', (room_id, iv["type"], iv["student"], iv["interviewer"],
              iv["program"], iv["cohort"], iv["date"], iv["date"],
              end_dt.strftime("%Y-%m-%d %H:%M:%S"), iv["duration"]))
        interview_id = cursor.lastrowid

        line_count = len(iv["transcript"])
        for j, (speaker, text, sent_label, emotions, engagement) in enumerate(iv["transcript"]):
            offset = timedelta(seconds=int(iv["duration"] * (j / max(line_count - 1, 1))))
            line_ts = (start_dt + offset).strftime("%Y-%m-%d %H:%M:%S")

            cursor.execute('INSERT INTO transcripts (interview_id, speaker, text, timestamp) VALUES (?, ?, ?, ?)',
                           (interview_id, speaker, text, line_ts))
            transcript_id = cursor.lastrowid
            total_transcripts += 1

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
            eng_level = "High" if engagement >= 8.0 else "Medium" if engagement >= 5.5 else "Low"

            words = text.split()
            keyphrases = []
            if len(words) > 3:
                for _ in range(min(3, len(words) - 1)):
                    idx = random.randint(0, len(words) - 2)
                    keyphrases.append(f"{words[idx]} {words[idx+1]}")

            cursor.execute('''
                INSERT INTO analysis_results (interview_id, transcript_id, sentiment_label, sentiment_confidence,
                    sentiment_positive, sentiment_neutral, sentiment_negative, emotions_json, keyphrases_json,
                    engagement_score, engagement_level, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (interview_id, transcript_id, sent_label, confidence, pos, neu, neg,
                  json.dumps(emotions), json.dumps(keyphrases), engagement, eng_level, line_ts))
            total_analysis += 1

        cursor.execute('''
            INSERT INTO interview_summary (interview_id, total_words, avg_sentiment_score,
                dominant_sentiment, dominant_emotion, top_topics_json, avg_engagement_score, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (interview_id, sum(len(t[1].split()) for t in iv["transcript"]),
              line_count, iv["dominant_sentiment"], iv["dominant_emotion"],
              json.dumps(iv["topics"]), iv["engagement_avg"], iv["date"]))

        topics = iv["topics"]
        cursor.execute('''
            INSERT INTO topic_classifications (interview_id, academics_percent, career_percent, faculty_percent,
                infrastructure_percent, mental_health_percent, social_percent, technology_percent,
                total_sentences, classified_sentences, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (interview_id, topics.get("Academics", 0), topics.get("Career", 0), topics.get("Faculty", 0),
              topics.get("Infrastructure", 0), topics.get("Mental health", 0), topics.get("Social", 0),
              topics.get("Technology", 0), line_count, line_count, iv["date"]))

        print(f"   [{i:2d}/15] {iv['student']:<25s} | {iv['type']:10s} | {iv['dominant_sentiment']:8s} | {iv['date'][:10]}")

    conn.commit()
    conn.close()

    print("-" * 60)
    print(f"   Added: 15 interviews, {total_transcripts} transcripts, {total_analysis} analysis rows")
    print("=" * 60)
    print("   Done! Total should now be 30 interviews.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    seed_batch2()
