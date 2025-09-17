import streamlit as st
import requests

st.title("🎬 Video Summarizer")

uploaded_video = st.file_uploader("Upload a video", type=["mp4"])

if uploaded_video is not None:
    with st.spinner("Processing video..."):
        try:
            files = {"video": (uploaded_video.name, uploaded_video.getvalue(), "video/mp4")}
            response = requests.post("http://localhost:8000/summarize_video/", files=files)

            if response.status_code == 200:
                result = response.json()
                text_output = result["result"]
                st.success("Video processed successfully!")

                st.download_button(
                    label="Download Summary",
                    data=text_output,
                    file_name="video_summary.txt",
                    mime="text/plain",
                )

                st.text_area("Summary", value=text_output, height=200)
            else:
                st.error(f"Error: {response.json()['error']}")
        except Exception as e:
            st.error(f"Error: {str(e)}")

