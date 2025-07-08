import streamlit as st
import pandas as pd
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid import GridUpdateMode, DataReturnMode
import time
import os
from dotenv import load_dotenv
from supabase import create_client
import backend as backend

# Load environment variables

load_dotenv()
url = os.getenv('SUPABASE_URL')
key = os.getenv('SUPABASE_KEY')
supabase = create_client(url, key)

# Model list

models = (
    "Course Similarity",
    "User Profile",
    "Clustering",
    "Clustering with PCA",
    "Neural Network",
    "Regression with Embedding Features",
    "Classification with Embedding Features"
)

# Load ratings from backend

def load_ratings():
    with st.spinner("Loading ratings from Supabase..."):
        return backend.load_rating()

# Load courses from backend

@st.cache_data
def load_courses():
    with st.spinner("Loading courses from Supabase..."):
        return backend.load_course()

# Load user-model map for a specific user

def load_user_model_map_by_userid(user_id):
    with st.spinner("Loading user model map from Supabase..."):
        return backend.load_user_model_map_by_userid(user_id)

# Load course selector UI using AgGrid

def course_selector(course_df):

    gb = GridOptionsBuilder.from_dataframe(course_df)
    gb.configure_default_column(enablePivot=True, enableValue=True, enableRowGroup=True)
    gb.configure_selection(selection_mode="multiple", use_checkbox=True)
    gb.configure_side_bar()
    grid_options = gb.build()

    response = AgGrid(
        course_df,
        gridOptions=grid_options,
        enable_enterprise_modules=True,
        update_mode=GridUpdateMode.MODEL_CHANGED,
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        fit_columns_on_grid_load=False,
    )

    results = pd.DataFrame(response["selected_rows"], columns=['COURSE_ID', 'TITLE', 'DESCRIPTION'])
    results = results[['COURSE_ID', 'TITLE']]
    return results

# Helper to refresh ratings when needed

def refresh_ratings():
    st.session_state['ratings_df'] = load_ratings()
    st.session_state['data_updated'] = False

# Streamlit page config

st.set_page_config(
    page_title="Course Recommender System",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🎓 Personalized Course Recommendation System")

# Sidebar options

st.sidebar.title('Personalized Learning Recommender')
existing_user = st.sidebar.selectbox('Are You An Existing User?', ['Yes', 'No'])

# Initial ratings loading

if 'ratings_df' not in st.session_state:
    refresh_ratings()

ratings_df = st.session_state['ratings_df']
course_df = load_courses()

# === NEW USER SECTION ===

if existing_user == 'No' and 'loaded_user' not in st.session_state:
    st.subheader("Select courses that you have completed:")
    selected_courses_df = course_selector(course_df)
    submit_new = st.button("Push to Database")

    if submit_new:
        if selected_courses_df.empty:
            st.warning("⚠️ Please select at least one course before submitting.")
        else:
            with st.spinner("Submitting your courses..."):
                new_user_id = pd.to_numeric(ratings_df['user'], errors='coerce').max() + 1
                new_rows = [
                    {"user": int(new_user_id), "item": course_id, "rating": 3}
                    for course_id in selected_courses_df['COURSE_ID']
                ]
                insert_response = supabase.table("Ratings").insert(new_rows).execute()

            if insert_response.data is not None:
                st.success(f"Successfully added ratings for new user ID {new_user_id}")
                st.session_state['loaded_user'] = new_user_id
                st.session_state['data_updated'] = True
                refresh_ratings()
                st.rerun()
            else:
                st.error("Failed to insert ratings into Supabase.")
                st.write(insert_response)

# === EXISTING USER SECTION ===

if existing_user == 'Yes' or 'loaded_user' in st.session_state:

    if existing_user == 'Yes':

        valid_user_ids = ratings_df['user'].unique()
        user_id = st.sidebar.number_input("Enter Your User ID", min_value=1, step=1)

        if user_id in valid_user_ids:
            if st.sidebar.button("Load My Data"):
                st.session_state['loaded_user'] = int(user_id)
        else:
            st.sidebar.warning("❌ Invalid User ID. Please enter a valid one.")        

    if 'loaded_user' in st.session_state:
        user_id = st.session_state['loaded_user']
        ratings_df = st.session_state['ratings_df']

        user_courses = ratings_df[ratings_df['user'] == user_id]
        enrolled_ids = user_courses['item'].unique()
        enrolled_courses = course_df[course_df['COURSE_ID'].isin(enrolled_ids)]

        st.subheader("📘 Courses you've already completed:")
        st.table(enrolled_courses[['COURSE_ID', 'TITLE']])

        selected_action = 'Model Options' if existing_user == 'No' else st.sidebar.radio("Choose Action:", ['Add Completed Courses', 'Model Options'])

        # === ADD ADDITIONAL COURSES ===

        if selected_action == 'Add Completed Courses':
            new_courses = course_df[~course_df['COURSE_ID'].isin(enrolled_ids)]
            st.subheader("Select additional completed courses:")
            selected_df = course_selector(new_courses)
            submit_add = st.button("Push Additional Courses")

            if submit_add:
                if selected_df.empty:
                    st.warning("⚠️ Please select at least one course before submitting.")
                else:
                    with st.spinner("Submitting..."):
                        new_rows = [
                            {"user": int(user_id), "item": course_id, "rating": 3}
                            for course_id in selected_df['COURSE_ID']
                        ]
                        insert_response = supabase.table("Ratings").insert(new_rows).execute()

                    if insert_response.data is not None:
                        delete_response = supabase.table("User_Model_Map").delete().eq("userid", user_id).execute()

                        if delete_response.data is not None:
                            st.info("✅ Existing trained models cleared.")

                            st.session_state['model_map_df'] = backend.load_user_model_map_by_userid(user_id)
                            st.session_state['last_loaded_user'] = user_id
                        else:
                            st.error("❌ Failed to clear trained models.")

                        st.success("Successfully added additional courses.")
                        st.session_state['data_updated'] = True
                        refresh_ratings()
                        st.rerun()
                    else:
                        st.error("Failed to insert additional ratings.")
                        st.write(insert_response)

        # === MODEL OPTIONS ===

        elif selected_action == 'Model Options':
            st.sidebar.markdown("---")
            st.sidebar.subheader("Model Selection")

            if (
                'model_map_df' not in st.session_state
                or st.session_state.get('data_updated', False)
                or st.session_state.get('last_loaded_user') != user_id
            ):
                st.session_state['model_map_df'] = load_user_model_map_by_userid(user_id)
                st.session_state['last_loaded_user'] = user_id
                st.session_state['data_updated'] = False

            model_map_df = st.session_state['model_map_df']
            trained_models = model_map_df['model'].tolist() if 'model' in model_map_df.columns else []
            untrained_models = [m for m in models if m not in trained_models]

            if trained_models:
                st.sidebar.markdown("**Trained Models**")
                selected_trained = st.sidebar.radio("Select Trained Model to Predict:", trained_models, key=f"trained_model_radio_{user_id}")
                if selected_trained and st.sidebar.button("Predict", key=f"predict_btn_{user_id}"):
                    with st.spinner(f"🔍 Predicting with {selected_trained}..."):
                        if selected_trained == "Course Similarity":
                            prediction_df = backend.course_similarity_predict(user_id)
                        elif selected_trained == "User Profile":
                            prediction_df = backend.user_profile_predict(user_id)
                        elif selected_trained == "Clustering" or selected_trained == "Clustering with PCA":
                            prediction_df = backend.kMeans_pred(user_id, selected_trained)
                        elif selected_trained == "Neural Network":
                            prediction_df = backend.NCF_predict(user_id)
                        elif selected_trained == "Regression with Embedding Features" or selected_trained == "Classification with Embedding Features":
                            prediction_df = backend.Embedding_Predict(user_id, selected_trained)
                        else:
                            prediction_df = pd.DataFrame()
                            st.warning(f"🚧 Prediction logic not implemented yet for {selected_trained}")

                    if not prediction_df.empty:
                        st.subheader("🎯 Recommended Courses:")
                        st.dataframe(prediction_df, use_container_width=True)
                    else:
                        st.info("No recommendations available or model not ready.")

            if untrained_models:
                st.sidebar.markdown("**Untrained Models**")
                selected_untrained = st.sidebar.radio("Select Untrained Model to Train:", untrained_models, key=f"untrained_model_radio_{user_id}")
                if selected_untrained and st.sidebar.button("Train", key=f"train_btn_{user_id}"):
                    placeholder = st.empty()
                    start_time = time.time()
                    with st.spinner(f"🔄 Training {selected_untrained}..."):
                        
                        if selected_untrained == "Course Similarity":
                            status = backend.course_similarity_train()
                        elif selected_untrained == "User Profile":
                            status = backend.user_profile_train()
                        elif selected_untrained == "Clustering" or selected_untrained == "Clustering with PCA":
                            status = backend.kMeans_train(selected_untrained)
                        elif selected_untrained == "Neural Network":
                            status = backend.NCF_train()
                        elif selected_untrained == "Regression with Embedding Features" or selected_untrained == "Classification with Embedding Features":
                            status = backend.Embedding_train(selected_untrained)
                        else:
                            status = f"🚧 Training logic not implemented yet for {selected_untrained}"

                    end_time = time.time()
                    total_seconds = round(end_time - start_time)
                    mins, secs = divmod(total_seconds, 60)
                    placeholder.success(f"{status} (⏱️ {mins} min {secs} sec)")
                    
                    if status.startswith("✅"):
                        supabase.table("User_Model_Map").insert({"userid": int(user_id), "model": selected_untrained}).execute()
                        st.session_state['data_updated'] = True
                        time.sleep(2)  
                        st.rerun()
                    else:
                        time.sleep(2)

        st.subheader("\U0001F3AF Use the sidebar to enter your courses, train your model, and view personalized recommendations.")
