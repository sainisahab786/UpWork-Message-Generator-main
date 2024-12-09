from flask import Flask, request, render_template
import google.generativeai as genai
import json
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


app = Flask(__name__)

# Define your project and skill data
projects = [
    {"name": "Object Detection with Detectron", "description": "Implemented object detection using the Detectron model on a custom dataset.", "skills": ["Python", "Detectron", "PyTorch", "Computer Vision"], "industry": "Technology"},
    {"name": "Image Classification with CNN", "description": "Built a CNN model using transfer learning with ResNet to classify images.", "skills": ["Python", "TensorFlow", "Keras", "CNN", "Transfer Learning"], "industry": "Technology"},
    {"name": "Customer Segmentation", "description": "Developed a predictive model for customer segmentation using clustering algorithms.", "skills": ["Python", "Scikit-learn", "Machine Learning", "Clustering"], "industry": "Retail"},
    {"name": "E-Book Word Frequency Tool", "description": "Developed a GUI-based tool that searches the Project Gutenberg library for eBooks, extracts the ten most frequent words from a given book title, and stores this information in a local SQLite database.", "skills": ["Python", "Tkinter", "SQLite", "Beautiful Soup", "NLTK", "Text Processing"], "industry": "Technology"},
    {"name": "AI Team", "description": "I have extensive experience in AI development, focusing on designing and deploying innovative solutions in natural language processing, computer vision, and predictive analytics. My roles have included creating scalable machine learning models, developing advanced neural networks, and optimizing IT operations through AI-powered automation. I have successfully led projects that bridge technical capabilities with business strategies, driving impactful results and contributing to organizational growth.", "skills": ["Natural Language Processing (NLP)","Computer Vision","Predictive Analytics","Machine Learning Model Development","Algorithm Optimization","Deep Learning","Reinforcement Learning","AI System Design","Software Engineering","Project Management","Research","Data Analysis","Deployment and Integration","AI Strategy","AIOps"], "industry": "Artificial Intelligence (AI) and related industries (e.g., technology, finance, healthcare, robotics)"}
]

skills = ["Python", "TensorFlow", "PyTorch", "Computer Vision", "Machine Learning", "Deep Learning", "Data Analysis", "Predictive Modeling", "Clustering", "Transfer Learning"]

# Initialize Gemini API (optional based on your existing setup)
def initialize_llm():
    genai.configure(api_key='AIzaSyCwccuIEZTRYt0AyD1EdNOo41soO8oY6CE')
    return genai.GenerativeModel(model_name='gemini-1.5-flash')

model = initialize_llm()

# Load the Sentence-BERT model
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Helper function for Cosine Similarity using TF-IDF
def calculate_cosine_similarity(text1, text2):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    similarity_matrix = cosine_similarity(tfidf_matrix)
    return similarity_matrix[0][1] * 100  # Return the similarity score in percentage

# Helper function for Sentence-BERT similarity
def calculate_sentence_similarity(text1, text2):
    embedding1 = sentence_model.encode(text1)
    embedding2 = sentence_model.encode(text2)
    similarity = cosine_similarity([embedding1], [embedding2])[0][0]
    return similarity * 100  # Similarity in percentage

# Extract project details from client requirements using the Gemini model
def extract_project_details(client_requirements):
    prompt = f"""
    Analyze the following project description to extract the following details:

    1. A brief description of the project.
    2. The industry to which the project belongs.
    3. The technology stack related to the project.
    4. The skills required to execute the project.
    5. The potential roles the user wants to fulfill.
    6. The pain points or challenges the client is likely facing.

    Project Description:
    "{client_requirements}"

    Output the result in a structured JSON format:
    {{
        "description": "...",
        "industry": "...",
        "technology": [...],
        "skills_required": [...],
        "roles": [...],
        "pain_points": "..."
    }}
    """
    
    response = model.generate_content(
        prompt,
        generation_config={'temperature': 0.7}
    )
    
    response_text = response.text.strip()
    if response_text.startswith('```json'):
        response_text = response_text[7:]  # Remove the first 7 characters ('```json')
    if response_text.endswith('```'):
        response_text = response_text[:-3]  # Remove the last 3 characters ('```')
    
    try:
        return json.loads(response_text)
    except json.JSONDecodeError as e:
        print("JSON Decode Error:", str(e))
        print("Response was:", response_text)
        return None

# Calculate match percentages using Cosine Similarity and Sentence-BERT
def calculate_percentage_match(extracted_info):
    percentages = {}

    # 1. Description match using Sentence-BERT for semantic similarity
    descriptions_similarity = [
        calculate_sentence_similarity(extracted_info['description'], project['description']) for project in projects
    ]
    percentages['description'] = max(descriptions_similarity)  # Take the best match

    # 2. Industry match using Cosine Similarity
    industry_similarities = [
        calculate_cosine_similarity(extracted_info['industry'], project.get('industry', ''))
        for project in projects if 'industry' in project
    ]
    percentages['industry'] = max(industry_similarities) if industry_similarities else 0

    # 3. Technology match using Cosine Similarity
    technologies = extracted_info.get('technology', [])
    if technologies:
        technology_similarities = [
            calculate_cosine_similarity(tech, ' '.join(project['skills']))
            for tech in technologies for project in projects
        ]
        percentages['technology'] = max(technology_similarities)
    else:
        percentages['technology'] = 0

    # 4. Skills match using Cosine Similarity
    extracted_skills = extracted_info.get('skills_required', [])
    if extracted_skills:
        skills_similarities = [
            calculate_cosine_similarity(skill, ' '.join(project['skills']))
            for skill in extracted_skills for project in projects
        ]
        percentages['skills'] = max(skills_similarities)
    else:
        percentages['skills'] = 0

    # 5. Roles match using Cosine Similarity
    roles = extracted_info.get('roles', [])
    if roles:
        roles_similarities = [
            calculate_cosine_similarity(role, ' '.join(project.get('roles', [])))
            for role in roles for project in projects
        ]
        percentages['roles'] = max(roles_similarities)
    else:
        percentages['roles'] = 0

    # 6. Pain points match using Cosine Similarity
    pain_points_keywords = ["challenges", "issues", "pain points"]
    pain_points_similarity = [
        calculate_cosine_similarity(extracted_info.get('pain_points', ''), keyword)
        for keyword in pain_points_keywords
    ]
    percentages['pain_points'] = max(pain_points_similarity) if pain_points_similarity else 0

    return percentages

# Generate client message based on project and skill data
def generate_client_message(extracted_info, projects, skills):
    prompt = f"""
    The client has provided the following project details:
    {json.dumps(extracted_info, indent=4)}

    You have the following data on past projects and skills:

    Projects:
    {json.dumps(projects, indent=4)}

    Skills:
    {json.dumps(skills, indent=4)}

Your task is to write a message to the client that is clear, professional, and easy to understand. The message should:

- Explain how your past experience is relevant to their project.
- Highlight the skills and technologies you have that can help with their project.
- Address any challenges or pain points they have mentioned and explain how you can solve them.
- Be written in simple, conversational English, with clear and well-organized paragraphs.
- Have a friendly, approachable, and confident tone.

The message should be structured in the following way:
1. Start with a greeting.
2. Briefly introduce yourself and mention your relevant experience.
3. Explain how your past projects are similar to their project and how your skills will help them.
4. Address the challenges or pain points they’ve mentioned and explain how you can solve them.
5. End with a confident, positive closing.

**Important Notes:**
- Use basic, easy-to-understand English, suitable for both technical and non-technical readers.
- Write in a natural, human-like way.
- Do not use "we"—only refer to yourself as "I".
- Avoid suggesting any direct connection to the client (e.g., no mentions of calls, meetings, or personal contact).

Please format the message as follows:

"Message to Client: ..."

    """
    response = model.generate_content(
        prompt,
        generation_config={'temperature': 0.7}
    )
    paragraphs = response.text.strip().split('\n')
    formatted_message = ''.join([f'<p>{para.strip()}</p>' for para in paragraphs if para.strip()])
    
    return formatted_message
    # print(response)
    # return response.text.strip()



# def generate_client_message(extracted_info, projects, skills):
#     prompt = f"""
#     The client has provided the following project details:
#     {json.dumps(extracted_info, indent=4)}

#     You have the following data on past projects and skills:

#     Projects:
#     {json.dumps(projects, indent=4)}

#     Skills:
#     {json.dumps(skills, indent=4)}

#     Your task is to generate a message for the client. The message should:
#     - Explain how your past project experience aligns with the client's project needs.
#     - Highlight the specific skills and technologies relevant to the project.
#     - Address the client's identified pain points and explain how you can solve them.
#     - Be professional, concise, and tailored to the client's requirements.
#     - Make sure the message is in very basic English and in human can understand
# has context menu

#     Provide the message in the following format:

#     "Message to Client: ..."
#     """
#     response = model.generate_content(
#         prompt,
#         generation_config={'temperature': 0.7}
#     )
#     return response.text.strip()














@app.route('/', methods=['GET', 'POST'])
def index():
    match_percentages = {}  # Initialize this variable to avoid undefined errors.

    if request.method == 'POST':
        action = request.form.get('action')
        client_requirements = request.form.get('client_requirements', '')
        extracted_info_str = request.form.get('extracted_info', '')

        if action == 'extract':
            extracted_info = extract_project_details(client_requirements)
            if extracted_info is None:
                return render_template('index.html', error="Failed to extract project details from LLM response.")
            
            # Calculate match percentages
            match_percentages = calculate_percentage_match(extracted_info)
            return render_template('index.html', client_requirements=client_requirements, extracted_info=extracted_info, match_percentages=match_percentages)

        if action == 'generate':
            extracted_info = json.loads(extracted_info_str)
            client_message = generate_client_message(extracted_info, projects, skills)
            # Calculate match percentages for the final message
            match_percentages = calculate_percentage_match(extracted_info)
            return render_template('index.html', client_message=client_message, extracted_info=extracted_info, match_percentages=match_percentages)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(port=8080, debug=True)
