import streamlit as st
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(page_title="SVM Guard", page_icon="🛡️")
st.title("🛡️ SecureMail SVM Filter")

emails = ["Win a free iPhone now", "Meeting at 11 am tomorrow", "Claim your prize immediately", "Project discussion with team", "Limited offer buy now"]
labels = [1, 0, 1, 0, 1]

vec = TfidfVectorizer(stop_words="english")
X_data = vec.fit_transform(emails)
svm_model = LinearSVC()
svm_model.fit(X_data, labels)

st.write("Enter the message content below for instant classification.")
user_input = st.text_area("Message Content", height=150, placeholder="Paste email text here...")

if st.button("Analyze Security"):
    if user_input.strip():
        processed_input = vec.transform([user_input])
        is_spam = svm_model.predict(processed_input)[0]
        if is_spam == 1:
            st.error("🚨 Classification: SPAM DETECTED")
        else:
            st.success("✅ Classification: CLEAN MESSAGE")
    else:
        st.warning("Input required.")
