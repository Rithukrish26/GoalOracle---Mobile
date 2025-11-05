import streamlit as st
import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import base64

st.set_page_config(page_title="GoalOracle ⚽", layout="centered")

# --- Helper functions ---
def calculate_score_probabilities(lambda_a, lambda_b, max_goals=8):
    matrix = np.zeros((max_goals + 1, max_goals + 1))
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            matrix[i, j] = poisson.pmf(i, lambda_a) * poisson.pmf(j, lambda_b)
    return matrix

def calculate_outcome_probabilities(prob_matrix):
    win_a = np.tril(prob_matrix, -1).sum()
    draw = np.trace(prob_matrix)
    win_b = np.triu(prob_matrix, 1).sum()
    return win_a, draw, win_b

def most_probable_score(prob_matrix):
    idx = np.unravel_index(np.argmax(prob_matrix), prob_matrix.shape)
    return idx, prob_matrix[idx]

# --- Custom CSS for mobile ---
st.markdown("""
<style>
.centered-logo {
    display: flex;
    justify-content: center;
    margin-top: -10px;
    margin-bottom: 10px;
}
input, .stNumberInput>div>div>input {
    max-width: 95% !important;
}
button[kind="primary"], button[kind="secondary"] {
    width: 90% !important;
    font-size: 18px !important;
    height: 50px !important;
    margin: 5px 0 !important;
}
button:hover {
    box-shadow: 0px 0px 15px rgba(0,208,192,0.7) !important;
    background-color: #00D0C0 !important;
    color: black !important;
}
</style>
""", unsafe_allow_html=True)

# --- Header Logo ---
logo = Image.open("Guluguluoracleaura.png")
scale_factor = 0.6  # slightly smaller for mobile
new_width = int(logo.width * scale_factor)
new_height = int(logo.height * scale_factor)
logo = logo.resize((new_width, new_height))

buffered = BytesIO()
logo.save(buffered, format="PNG")
encoded_logo = base64.b64encode(buffered.getvalue()).decode()

st.markdown(f"""
<div class="centered-logo">
    <img src="data:image/png;base64,{encoded_logo}" width="{new_width}" height="{new_height}">
</div>
""", unsafe_allow_html=True)

# --- Inputs stacked vertically ---
st.subheader("Team A — Inputs")
ta_goals = st.number_input("Goals Scored (λ)", min_value=0.0, value=1.2, step=0.1, format="%.2f")
ta_conceded = st.number_input("Goals Conceded", min_value=0.0, value=1.0, step=0.1)
ta_sot = st.number_input("Shots on Target", min_value=0.0, value=3.0, step=0.1)
ta_chances = st.number_input("Chances Created", min_value=0.0, value=5.0, step=0.1)
ta_poss = st.number_input("Possession (%)", min_value=0.0, max_value=100.0, value=52.0, step=0.1)
ta_pass = st.number_input("Pass Completion (%)", min_value=0.0, max_value=100.0, value=82.0, step=0.1)

st.subheader("Team B — Inputs")
tb_goals = st.number_input("Goals Scored (λ)", min_value=0.0, value=1.0, step=0.1, format="%.2f")
tb_conceded = st.number_input("Goals Conceded", min_value=0.0, value=1.1, step=0.1)
tb_sot = st.number_input("Shots on Target", min_value=0.0, value=2.7, step=0.1)
tb_chances = st.number_input("Chances Created", min_value=0.0, value=4.0, step=0.1)
tb_poss = st.number_input("Possession (%)", min_value=0.0, max_value=100.0, value=48.0, step=0.1)
tb_pass = st.number_input("Pass Completion (%)", min_value=0.0, max_value=100.0, value=79.0, step=0.1)

# --- Buttons ---
col1, col2 = st.columns(2)
with col1:
    predict = st.button("Predict")
with col2:
    reset = st.button("Reset")

# --- Logic ---
if reset:
    st.experimental_rerun()

if predict:
    lambda_a = ta_goals
    lambda_b = tb_goals
    prob_matrix = calculate_score_probabilities(lambda_a, lambda_b)
    win_a, draw, win_b = calculate_outcome_probabilities(prob_matrix)
    (best_i, best_j), best_p = most_probable_score(prob_matrix)

    st.subheader("Prediction Results")
    st.write(f"Most Probable Score: {best_i} - {best_j} ({best_p:.2%})")
    st.write(f"Team A Win: {win_a:.2%} | Draw: {draw:.2%} | Team B Win: {win_b:.2%}")

    # Heatmap
    fig, ax = plt.subplots()
    im = ax.imshow(prob_matrix, origin='lower', aspect='auto')
    ax.set_xlabel('Team B Goals')
    ax.set_ylabel('Team A Goals')
    ax.set_title('Score Probability Matrix')
    fig.colorbar(im, ax=ax)
    st.pyplot(fig)

st.markdown("[Visit GoalOracle GitHub](https://github.com/Rithukrish26/GoalOracle---Streamlit/tree/main)")
