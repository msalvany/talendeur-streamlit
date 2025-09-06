import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import io as io
import base64
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
import string
import re
from pathlib import Path
import unicodedata
import re
from fuzzywuzzy import process
from sentence_transformers import SentenceTransformer, util
from flashtext import KeywordProcessor
import torch


#Page configuration

#Import banner image

def get_img_as_base64(img_path: str) -> str:
    #reads the image file and encodes it to base64
    #Note:this is more portable than using the image directly, as it avoids issues with file paths in different environments.
    #we are converting the string to a Path object to ensure compatibility with different operating systems
    img_path = Path(img_path)
    with img_path.open("rb") as image_file:
        data = image_file.read()
    return base64.b64encode(data).decode("utf-8")

img_banner = get_img_as_base64("Banner_Streamlit.png")

st.markdown(f"""
    <style>
    .fixed-banner {{
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        z-index: 9999;
        background-color: white;
        box-shadow: 0px 2px 6px rgba(0, 0, 0, 0.1);
    }}

    .fixed-banner img {{
        width: 100%;
        height: auto;
        display: block;
    }}

    /* Add cushion or space between the banner and your text*/
    .stApp {{
        padding-top: 150px; /* Adjust the height of your banner */
    }}
    </style>

    <div class="fixed-banner">
        <img src="data:image/png;base64,{img_banner}" alt="Talendeur company banner." >
    </div>
""", unsafe_allow_html=True)




st.set_page_config(page_title="Candidate Analysis", layout="wide")
st.title("Candidate Analysis Dashboard")

#Load the data
# Note: Ensure the path to your CSV file is correct.
#data_path=Path("profile_streamlit_8.csv")

#df = pd.read_csv(data_path, encoding='utf-8', low_memory=False)

# Assign df to df_candidate for further use
#df_candidate = df
# Display the first few rows of the dataframe
#st.write("Data loaded successfully. Here are the first few rows of the dataframe:")
#st.dataframe(df_candidate.head())"
##New code to incorporate for adding more than one file
# ...existing code...

# 1. File upload and dataframe creation
work_df = pd.read_csv("work_with_id.csv")
edu_df = pd.read_csv("education_with_id.csv")
cert_df = pd.read_csv("certifications_with_id.csv")
ref_df = pd.read_csv("df_references_with_id.csv")
vol_df = pd.read_csv("df_volunteer_experiences_with_cause_2.csv")
int_df = pd.read_csv("international_experience_2.csv")
df_candidate = pd.read_csv("profile_streamlit_8.csv")


# 2. Candidate ID extraction
candidate_ids = []

if work_df is not None:
    candidate_ids = work_df["candidate_id"].unique().tolist()
elif edu_df is not None:
    candidate_ids = edu_df["candidate_id"].unique().tolist()
elif cert_df is not None:
    candidate_ids = cert_df["candidate_id"].unique().tolist()
elif ref_df is not None:
    candidate_ids = ref_df["candidate_id"].unique().tolist()
elif df_candidate is not None:
    candidate_ids = df_candidate["candidate_id"].unique().tolist()
elif int_df is not None:
    candidate_ids = int_df["candidate_id"].unique().tolist()
elif vol_df is not None:
    candidate_ids = vol_df["candidate_id"].unique().tolist()

# 3. Sidebar setup and candidate selection
st.sidebar.title("Candidate Selection")
st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] {
            background-color: #C3CDE6;
        }
        section[data-testid="stSidebar"] .css-1d391kg {
            color: white;
        }
    </style>
    """,
    unsafe_allow_html=True
)

if candidate_ids:
    selected_id = st.sidebar.selectbox("Select a candidate", candidate_ids)
else:
    st.sidebar.warning("No candidates available. Please upload data files.")
    selected_id = None

# 4. Filter dataframes based on selected candidate
if selected_id is not None:
    work_filtered = work_df[work_df["candidate_id"] == selected_id] if work_df is not None else None
    edu_filtered = edu_df[edu_df["candidate_id"] == selected_id] if edu_df is not None else None
    cert_filtered = cert_df[cert_df["candidate_id"] == selected_id] if cert_df is not None else None
    ref_filtered = ref_df[ref_df["candidate_id"] == selected_id] if ref_df is not None else None
    int_filtered = int_df[int_df["candidate_id"] == selected_id] if int_df is not None else None
    df_candidate_selected = df_candidate[df_candidate["candidate_id"] == selected_id] if df_candidate is not None else None
else:
    work_filtered = edu_filtered = cert_filtered = ref_filtered = df_candidate_selected = None



#Create a word cloud for the candidate's biography
st.header("Candidate's Bio")

nltk.download('stopwords')
#set stopwords to English
stop_words = set(stopwords.words('english'))
#Add some custom stopwords
#These are repetitive words that do not add value to the word cloud but are not in the nltk stopwords list.
custom_stopwords = {'candidate', 'profile', 'experience', 'years', 'work', 'job', 'skills', 'knowledge'}
stop_words.update(custom_stopwords)
all_stopwords = stop_words.union(set(custom_stopwords))

#define a function to clean the text
def clean_text(text):
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Convert to lowercase
    text = text.lower()
    # Remove stopwords
    text = ' '.join([word for word in text.split() if word not in all_stopwords])
    return text
#Apply the clean_text function to the biography column of the selected candidate
cleaned_text=clean_text(df_candidate_selected.iloc[0]['bio'])
# Generate the word cloud
wordcloud = WordCloud(
    width=600,
    height=300,
    background_color='white',
    colormap='twilight',
    stopwords=all_stopwords,
    min_font_size=10
).generate(cleaned_text)

# Display the word cloud
fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis('off')  # Hide the axes
st.pyplot(fig)



#
#Key metrics
st.header("Key Metrics")
# Display key metrics for the selected candidate
col1, col2,col3= st.columns(3)
col1.metric("Top_degree", df_candidate_selected.iloc[0]["highest_qualification"])
col2.metric("Total experience (years)",round( df_candidate_selected.iloc[0]["total_years_experience"]))
col3.metric("Avg_years", int(df_candidate_selected.iloc[0]["avg_years_per_job"]))




#Chosing most relevant areas of knowledge to show in the rose radial chart
#Dictionaries
#Creating dictionaries associating job_posts  with different dimensions

dimensions_work={

    #Creativity

    'creativity':['designer','graphic designer','ux','ux/ui','art director','art director','content creator','illustrator','copywriter','editor','graphic artist',
                  'brand manager','brand lead','brand strategist', 'social media manager','social media strategist','social media coordinator',
                  'tv producer','marketing specialist','marketing executive','marketing manager', 'marketing analyst', 'product manager','product owner',
                  'product strategist', 'entrepreneur','start-up founder', 'film director','sound designer', 'audio engineer', 'visual effects', 'video editor'
                  'video director', 'photographer', 'cinematographer','cinematographer','chef','cook','sous-chef','line cook','musician', 'composer','actor',
                  'actress','director','writer', 'game designer', 'motion graphics designer','architect', 'interior designer', 'industrial designer','teacher',
                  'elementary school teacher','high school teacher', 'middle school teacher','secondary school teacher','primary school teacher','professor', 'publisher',
                  'journalist','reporter','fashion designer', 'fashion director','music producer', 'music engineer','tv writer', 'technical writer', 'stylist', 'senior stylist',
                  'cosmetologist','make-up artist','makeup artist','r&d innovation specialist','r&d researcher', 'r&d engineer', 'education innovation specialist', 'education researcher',
                  'education researcher'],

    #Communication

    'communication': [ 'journalist', 'public relations specialist','communications officer','communications manager','corporate communications director','media relations specialist',
                        'spokeperson','marketing communications specialist','brand comunications manager','content writer','copywriter', 'social mmedia manager','community manager',
                        'editor', 'media producer','tv host','podcast host','news anchor','broadcaster','customer services representative','customer service manager',
                        'client relationship representative','client relationship manager','sales representative', 'sales manager','human resources specialist', 'human resources manager',
                        'recruiter', 'recruiter manager','hr specialist','hr manager','teacher','elementary school teacher','high school teacher','middle school teacher','secondary school teacher',
                        'primary school teacher','professor','publisher','reporter','tv writer','technical writer','stylist','lecturer','senior lecturer','corporate trainer','workshop facilitator',
                        'workshop trainer','communication coach','speach coach','event planner','event manager','interpreter','translator','marketing coordinator','marketing manager', 'marketing analyst',
                        'marketing director', 'customer sucess coordinator','customer success manager', 'documentation manager','documentation specialist', 'senior event manager','principal research scientist',
                        'research scientist','researcher','senior research scientist','research analyst', 'education director','associate attorney','senior attorney','attorney','event coordinator','communication manager',
                        'communication specialist', 'cosmetologist','make-up artist','director of clinical research','clinical research specialist','clinical research manager','curriculum cordinator','librarian',
                        'library assistant','library specialist','pr coordinator','education innovation specialist',
                        'education researcher','design thinking facilitator'],


    #critical thinking
    'critical thinking': ['analyst', 'research analyst','research scientist','researcher','senior research scientist', 'senior ai researcher','business analyst','business intelligence analyst',
                          'data analyst','data scientist','data engineer','education director','associate attorney','senior attorney', 'attorney','market research analyst','director of clinical research',
                          'clinical research specialist','clinical research manager','curriculum cordinator','librarian','library assistant','library specialist','education innovation specialist',
                          'education researcher','design thinking facilitator', 'software engineer','software developer', 'senior software engineer','senior software developer', 'financial analyst',
                          'senior financial analyst','financial manager','senior financial manager','accounting analyst','senior accounting analyst','accounting manager','senior accounting manager',
                          'senior devops engineer','devops engineer','devops specialist','devops manager','senior devops manager','product manager','product owner','innovation manager',
                          'product strategist','entrepreneur','start-up founder','senior accountant','staff accountant','insurance underwriter','senior insurance underwriter','insurance manager',
                          'senior insurance manager','finance director','senior finance director','accounting director','senior accounting director'],
    #Innovation
        'innovation': ['product manager','product owner','innovation manager','product strategist','entrepreneur','start-up founder',
                        'vice-president of innovation','director of innovation','chief innovation officer','innovation strategist',
                        'new product development manager','new product development specialist','product development manager','product development specialist',
                        'business innovation lead','r&d innovation specialist','r&d researcher','r&d engineer','senior r&d engineer','senior r&d researcher',
                        'technolgy innovation manager','technolgy innovation specialist','technolgy innovation engineer','open innovation manager','open innovation specialist',
                        'open innovation engineer','digital innovation manager','digital innovation specialist','digital marketing manager','digital marketing specialist',
                        'sustainability innovation lead','education innovation specialist','design thinking facilitator','process improvement engineer',
                        'process improvement manager','senior process improvement engineer','senior process improvement manager' ],

    # technology & development

    'technology & development': ['software engineer', 'software developer','senior software engineer','senior software developer', 'full stack developer','front-end developer',
                                  'back-end developer','devops engineer','devops specialist','devops manager','lead developer','senior devops engineer','senior devops manager',
                                  'senior lead developer','data engineer','data scientist', 'data analyst','ux designer', 'ux/ui','it support specialist','it support manager',
                                  'ux/ui designer', 'senior ux designer','senior ux/ui designer','cloud solutions architect','cloud solutions engineer','cloud solutions manager',
                                  'senior cloud solutions architect','senior cloud solutions engineer','principal cloud architect','cloud architect','cloud engineer',
                                  'cybersecurity analyst','cybersecurity engineer','cybersecurity manager','senior cybersecurity analyst','senior cybersecurity engineer',
                                  'senior cybersecurity manager','network engineer', 'network manager','senior network engineer','senior network manager','mobile appp developper',
                                  'mobile app developer','senior mobile app developer','ios developer','android developer','senior ios developer','senior android developer','web developer',
                                  'senior web developer','QA tester','QA engineer','QA manager','senior QA tester','senior QA engineer','senior QA manager','test automation engineer',
                                   'test automation manager','senior test automation engineer','senior test automation manager','data architect', 'data administrator','senior data architect',
                                   'senior data administrator'],


    #Operations
    'operations': ['operations analyst','operations manager','operations specialist','operations coordinator','operations executive','operations director','supply chain analyst','supply chain manager',
                   'supply chain specialist','supply chain coordinator','supply chain executive','supply chain director','logistic analyst','logistic manager','logistic specialist','logistic coordinator',
                   'logistic executive','logistic director','warehouse manager','warehouse specialist','warehouse coordinator','warehouse director','event operations manager','event operations specialist',
                    'event operations coordinator','event operations executive','event operations director','event coordinator','construction supervisor','construction manager', 'construction engineer',
                    'construction coordinator','construction executive','construction director','general contractor','contractor','truck driver','fleet safety coordinator','senior process improvement engineer',
                    'fleet safety manager','fleet safety specialist', 'fleet safety executive','fleet safety director', 'transportation coordinator','transportation manager','branch operations executive',
                    'transportation specialist', 'transportation executive','transportation director','process improvement engineer','process improvement manager','branch operations director',
                    'process improvement coordinator','process improvement executive','process improvement director','branch operations supervisor','branch operations manager','branch operations coordinator'
                    ],

    #Social Impact
    'social impact': ['social impact analyst','social impact manager','social impact specialist','social impact coordinator','social impact executive','social impact director',
                      'senior environmental engineer','environmental engineer','environmental manager','staff nurse','nurse','senior nurse','nurse practitioner','social worker',
                      'clinical social worker','social worker coordinator','social worker director','director of social services','police officer','detective', 'senior detective',
                      'detective sergeant','senior detective sergeant','librarian','firefighter','fire lieutenant','fire officer','fire captain','dental hygienist','senior dental hygienist',
                      'dental hygienist assistant','senior director of social services' ],


    #Business Acumen
    'business acumen': ['business analyst','business intelligence analyst','data analyst','data scientist','senior data scientist','senior data analyst',
                        'entrepreneur','start-up founder','senior accountant','staff accountant','insurance underwriter','senior insurance underwriter',
                        'insurance manager','senior insurance manager','finance director','senior finance director', 'accounting director','senior accounting director',
                        'accounting analyst','senior accounting analyst','accounting manager','senior accounting manager','real estate agent','commercial real estate broker',
                        'real estate investment manager','real estate broker','real estate manager','bank teller','bank manager','banker','senior banker','senior underwriter',
                        'underwriting manager','junior accountant','financial analyst','senior financial analyst','finance manager','senior finance manager','financial manager',
                        'senior financial manager']



             }

#Normalization function
#Create a function to normalize the job titles
def normalize_text(text,*,collapse_spaces=True,remove_accents=True,replace_hypen_with_space=True,unify_ampersand=True):
   if pd.isna(text):
        return None
   text = str(text).strip().lower()
   if remove_accents:
      text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
   #even though there are no accents in english, a few foreign words might appear
   # also useful for multilanguage model
   if replace_hypen_with_space:
    text = text.replace('-', ' ')
    # changes front-end to front end
   if unify_ampersand:
    text = text.replace('&', 'and')
   if collapse_spaces:
    text = ' '.join(text.split())
    # remove points, for example Sr. for senior
    text = re.sub(r'\b(sr|jr)\.\b', r'\1', text)
   return text

#Set the first model to use for embeddings
model_transformer_1 = SentenceTransformer('distiluse-base-multilingual-cased-v2')
#using a multilanguage model taking in to consideration scalability to other languages
#Set the embeddings for each dimension
#We are using the keywords defined in the dimensions_work dictionary to create the embeddings for each dimension
skill_embeddings = {
    skill: model_transformer_1.encode(keywords, convert_to_tensor=True)
    for skill, keywords in dimensions_work.items()
}
def map_title_to_skills(title, skill_embeddings, threshold=0.85):
    title_emb = model_transformer_1.encode(normalize_text(title), convert_to_tensor=True)
    results = {skill: 0 for skill in skill_embeddings.keys()}
    for skill, emb_list in skill_embeddings.items():
        cos_scores = util.cos_sim(title_emb, emb_list)
        if cos_scores.max().item() >= threshold:
            results[skill] += 1
    return results
#Apply the function to the work_df dataframe
#Here we use the full matrix of job titles to get the skills for each job title
df_work_skills = work_df['job_title'].apply(map_title_to_skills, args=(skill_embeddings,))
df_work_skills = pd.DataFrame(list(df_work_skills))
df_work_skills['candidate_id'] = work_df['candidate_id']

#Add by candidate_id
df_skills_final = df_work_skills.groupby('candidate_id').sum().reset_index()

# Normalize column names
df_skills_final.columns = df_skills_final.columns.str.replace(' & ', '&').str.lower()
df_skills_final.rename(columns={'candidate_id': 'candidate_id'}, inplace=True)

#Creating dictionaries associating words in references with different dimensions
dimensions_ref={

    #Business Acumen
    'business acumen':['business sense','business acumen','project management skills','strategic vision','strategic vision','vendor relationships','customer satisfaction',
                       'business expertise','actionable insights','stakeholder management abilities','risk management expertise','crucial insights'],
    #Collaboration
    'collaboration':['collaborative approach','collaborated','staff development','compassionate','mentorship','coordinate',
                      'collaborative spirit','mentoring approach','collaborative mindset','collaborative leadership',
                      'cross-functional teams','willingness to mentor','team player','collaboration','diplomacy','cross-functional leadership skills'],
    #Leadership
    'leadership': ['problem-solving skills','collaborative approach','decision-making abilities','staff development','mentorship','strategic vision',
                    'staff supervision','crisis situations','program development', 'program management','handle challenging situations','leadership abilities',
                    'coordinate','problem-solving abilities', 'showed initiative','staff coordination','campaign execution','collaborative leadership',
                    'stakeholder management abilities','proactive approach','ability to troubleshoot','cross-functional leadership skills','execution capabilities'],
    #Innovation
    'innovation':['innovative','trends','designed','creative approach','process improvement','digital trends','innovative thinking','creative vision','creative flair',
                  'digital transformation','creative','innovative approach'],
    #Precision
     'precision': ['attention to detail','detailed','diagnostic skills','thorough','expertise','investigative abilities','meticulous','organizational skills',
                    'optimization','efficiency','efficient','accurate','well-documented','proficient','high-quality','commitment to excellence','research capabilities',
                    'rigorous','rigorous methodology','analytical rigor','accuracy','optimize performance','methodical','data-driven','well-crafted','effective',
                    'lean processes'],

    #Critical Thinking
     'critical thinking': ['analytical abilities','problem-solving skills','assessment skills','research skills','diagnostic skills','decision-making abilities',
                            'strategic vision','crisis situations','program development','program management','investigative abilities','organizational skills',
                            'optimization','problem-solving abilities','learning capacity','well-documented','complex problems','analytical mindset','translate complex data',
                            'research capabilities','intellectual curiosity','analytical rigor', 'strategic thinking','actionable insights','risk management expertise',
                            'crucial insights','ability to troubleshoot','data-driven','analytical approach'],
    #depth of work
    'depth':['comprehensive','complex','knowledge','research skills','diagnostic skills','thorough','commitment','expertise','investigative abilities','optimization',
             'learning capacity','well-documented','complex problems','high-quality','translate complex data','deep','commitment to excellence','research capabilities',
             'complex experiments','intellectual curiosity','analytical rigor','thoughtful','complex projects','risk management expertise','data-driven'],
    #commitment
    'commitment':['dedication','commitment','staff development','mentorship','consistently','learning capacity','genuine passion','passion','reliable','trusted',
                  'confidence','reliability','enthusiasm','continuous learning'],
    #social impact
    'social impact':['advocacy skills','community participation','education','mentorship','funding','program development','mentoring approach'],

    #communication
    'communication': ['advocacy skills','client communication','education','staff development', 'mentorship','staff supervision','program development','community relationships',
                      'vendor relationships','organizational skills','coordinate','client relationships','communication skills','staff coordination','customer satisfaction',
                      'communicate solutions','collaborative mindset','cross-functional teams','actionable insights','stakeholder management abilities','diplomacy','ability to communicate',
                      'customer service skills','ability to explain','interpersonal skills','cross-team'],
    #empathy
    'empathy':['compassionate','mentorship','mentoring approach','constructive','team player','diplomacy','customer service skills','interpersonal skills'],

    #flexibility
    'flexibility':['adapt','cross-functional teams','versatility','ability to adapt','ability to incorporate client feedback','ability to incorporate feedback','team player',
                   'stakeholder management abilities','diplomacy','customer service skills','interpersonal skills','cross-team','adaptability','cross-functional leadership skills']



                                    }
#Normalize the dictionary dimensions_ref
dimensions_ref_norm = {
    skill: [normalize_text(kw) for kw in keywords]
    for skill, keywords in dimensions_ref.items()
}
#Normalize the column ref_text in the matrix df_references
ref_df['ref_text']=ref_df['ref_text'].apply(normalize_text)

#KeywordProcessor
from flashtext import KeywordProcessor
from sentence_transformers import SentenceTransformer, util
kp=KeywordProcessor()
for ref,words in dimensions_ref_norm.items():
  #adds all the keywords list
  #kp.add_keywords_from_list(words,ref) # This line is causing the error
  #each word is included individually
  for word in words:
    kp.add_keyword(word,ref)

#Embeddings model
model_transformer=SentenceTransformer('distiluse-base-multilingual-cased-v2')

def build_ref_embeddings(dimensions_ref_norm,model_transformer):
  ref_embeddings={}
  for ref,words in dimensions_ref_norm.items():
    embedding_list=[model_transformer.encode(word,convert_to_tensor=True)for word in words]
    ref_embeddings[ref]=embedding_list
  return ref_embeddings

ref_embeddings=build_ref_embeddings(dimensions_ref_norm,model_transformer)

import torch

def analyze_ref_text(ref_text,model_transformer,threshold=0.85):
    # Initialize results to store counts for each dimension
    results = {ref: 0 for ref in dimensions_ref_norm.keys()}

    # ---- A. Exact keyword matching ----
    found_exact = kp.extract_keywords(ref_text)
    for ref in found_exact:
        results[ref] += 1

    # ---- B. Semantic matches con embeddings ----
    text_emb = model_transformer.encode(ref_text, convert_to_tensor=True)
    for ref, emb_list in ref_embeddings.items():
        emb_tensor = torch.stack(emb_list)

        cos_scores = util.cos_sim(text_emb, emb_tensor) # shape [1, number of keywords]

        # Check if any of the similarity scores for keywords in this dimension meet the threshold
        if cos_scores.max().item() >= threshold:
             # If at least one keyword meets the threshold, increment the count for the dimension
            results[ref] += 1

    return results

#Apply analyze reference over normalized text

ref_results=ref_df['ref_text'].apply(analyze_ref_text, args=(model_transformer,))
#Convert in to a dataframe
df_ref_results=pd.DataFrame(list(ref_results))
#Add candidate_id
df_ref_results['candidate_id']=ref_df['candidate_id']

#Group by candidate id all references
df_ref_results=df_ref_results.groupby('candidate_id').sum().reset_index()

# Normalize column names
df_ref_results.columns = df_ref_results.columns.str.replace(' & ', '&').str.lower()
df_ref_results.rename(columns={'candidate_id': 'candidate_id'}, inplace=True)

#Merge the two dataframes with dimensions or skills, the one we created from the job experiences and the one from the references

def merge_work_ref_skills(df1=df_skills_final,df2=df_ref_results,id_col='candidate_id'):
#Create an outer join, identify columns with suffixes
  df_merge = pd.merge(df1, df2, on=id_col, how="outer", suffixes=("_1", "_2"))
#we identify the common columns , excluding the id
  common_cols = set(df1.columns).intersection(set(df2.columns))
  common_cols.discard(id_col)
  #note with set we use discard instead of remove

  #We add the common columns and we eliminate the duplicates
  for col in common_cols:
    df_merge[col] = df_merge[f"{col}_1"].fillna(0) + df_merge[f"{col}_2"].fillna(0)
    df_merge.drop([f"{col}_1", f"{col}_2"], axis=1, inplace=True)

  #Fill in the possible Nan with zero
  df_merge=df_merge.fillna(0)

  return df_merge

df_skills_final = df_work_skills.groupby('candidate_id').sum().reset_index()


df_ref_results = df_ref_results.groupby('candidate_id').sum().reset_index()


#Apply merge_work_ref_skills
df_final=merge_work_ref_skills(df1=df_skills_final,df2=df_ref_results,id_col='candidate_id')




df_final_selected = df_final[df_final['candidate_id'] == selected_id]



st.header("Skills and Aptitudes by dimension")



# List of dimensions (columns in df_final)
dimensions_columns = [
    "collaboration", "leadership", "precision", "depth", "commitment", "empathy", "flexibility",
    "critical thinking", "technology&development", "operations", "social impact", "business acumen",
    "creativity", "innovation", "communication"
]

# Filter for selected candidate
candidate_row = df_final[df_final['candidate_id'] == selected_id]

if candidate_row.empty:
    st.warning("No data available for the selected candidate.")
else:
    # Get values for each dimension (fill missing with 0)
    values = [candidate_row[col].values[0] if col in candidate_row.columns else 0 for col in dimensions_columns]



    # Build DataFrame for chart
df_polar = pd.DataFrame({
        "Dimension": dimensions_columns,
        "Value": values
    })

  
candidate_row = df_final[df_final['candidate_id'] == selected_id]


if candidate_row.empty:
    st.warning("No data available for the selected candidate.")
else:
   
    values = [candidate_row[col].values[0] if col in candidate_row.columns else 0 for col in dimensions_columns]

    df_polar = pd.DataFrame({
        "Dimension": dimensions_columns,
        "Value": values
    })
   

    if df_polar.empty or df_polar["Value"].sum() == 0:
        st.warning("No dimensions with value > 0 for this candidate.")
    else:

    # Plot polar chart
        fig = px.bar_polar(df_polar, r="Value", theta="Dimension",
                       color="Value", color_continuous_scale=px.colors.sequential.Sunset)
        fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=False,
        title="Dimensions of Knowledge & Expertise",
        width=900,
        height=700
    )
st.plotly_chart(fig, use_container_width=True)
st.markdown(" The values represent the level of expertise in each area.")



#Create a bar chart for the top 5 certifications areas
st.header("Top Certifications Areas")
#define the columns to use
#keeping only the columns that start with 'cert_'
cert_columns = [col for col in df_candidate_selected.columns if col.startswith('cert_')]

#define the values to use
#creates a Series with the values of the certifications for the selected candidate
# We assume that the first row contains the values for the selected candidate
cert_values = df_candidate_selected[cert_columns].iloc[0]
#In cert_df, 'values is the numeric column with the number of certifications.
#The index is the column with the certification names.
cert_df=(cert_values.to_frame(name='values').reset_index(names='cert_name'))
#Clean the certification names for better readability
cert_df['cert_name'] = cert_df['cert_name'].str.replace('cert_', '',regex=False).str.replace('_', ' ',regex=False).str.title()


# Create a bar chart function
def create_bar_chart(cert_columns, cert_values,candidate_id=selected_id):
    #create df and clean title columns
    df_cert_bar = pd.DataFrame({'Certifications': cert_values.values},index=[col.replace('cert_', '').replace('_', ' ').title() for col in cert_columns])                        
   
   #Modify the index to be more readable
    df_cert_bar.reset_index(inplace=True)
    #Rename the index column to 'Category' for clarity
    df_cert_bar.rename(columns={'index': 'Category'}, inplace=True)   
    #Keep only the top 5 certifications
    #First segment the positive values
    df_cert_bar = df_cert_bar[df_cert_bar['Certifications'] > 0]
    #Sum the values by category
    total_cert=df_cert_bar["Certifications"].sum()
    #If there are no certifications, we don't want to divide by zero
    if total_cert == 0:
        st.warning("No certifications found for the selected candidate.")
        return None
    #Calculate the percentage of each certification
    df_cert_bar['Percentage'] = ((df_cert_bar['Certifications'] / total_cert) * 100).round(2)
    #Sort the values by percentage and keep only the top 5
    df_cert_bar = df_cert_bar.nlargest(5, 'Certifications').sort_values(by='Certifications', ascending=True)
    # Create the bar chart showing the top 5 certifications and their percentages
    #Note: we are using the 'Percentage' column to show the percentage of each certification
    #Note:we are inverting 'x' and 'y' to have the categories on the y-axis and the percentages on the x-axis
    fig = px.bar(
        df_cert_bar,
        x='Certifications',
        y='Category',
        text='Percentage',
        orientation='h',
        title=f'Top Certifications Areas for Candidate {candidate_id}',
        labels={'Percentage': 'Porcentaje (%)', 'Category': 'Categoría'},
         color='Certifications',
        color_continuous_scale=px.colors.sequential.Sunset,
    )
    # Update the layout of the chart
    fig.update_traces(texttemplate='%{y}<br>%{text:.2f}%', textposition='inside')
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    fig.update_layout(showlegend=False)
    fig.update_xaxes(rangemode="tozero")
    return fig
#Let's call the function to create the bar chart
fig_certificaciones = create_bar_chart(cert_columns, cert_values)
if fig_certificaciones:
    st.plotly_chart(fig_certificaciones, use_container_width=True)
    st.markdown("""This chart shows the top 5 certification areas for the candidate, along with their respective percentages.
                If the candidate a condensed profile it may show fewer areas.""")  
else:
    st.markdown("""No certifications found for the selected candidate.""")


#Volunteering experience
st.header("Volunteering Experience")
#Check if there is volunteering experience for the selected candidate

#Estructuring the data for the volunteering experience chart
#Processing time sensitive data
def process_date(date_str):
    df = pd.DataFrame(vol_df[vol_df['candidate_id'] == selected_id])
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["end_date"] = pd.to_datetime(df["end_date"], errors='coerce')
    df["duration_days"] = (df["end_date"] - df["start_date"]).dt.days
    df["duration_months"] = df["duration_days"] / 30.44  # Approximate conversion to months
    df["start"] = df["start_date"]
    df["end"] = df["end_date"]
    return df
vol_candidate = process_date(selected_id)


st.subheader("Causes Supported")
cause_summary = vol_candidate.groupby("cause")["duration_days"].sum().reset_index()
custom_palette = [
    "#E97DF5", 
    "#8E44AD", 
    "#BB8FCE", 
    "#AF5495", 
    "#BB138B", ]
    

fig_donut = px.pie(
    cause_summary,
    names="cause",
    values="duration_days",
    hole=0.4,
    title="Time invested by Cause",
    color_discrete_sequence=custom_palette,
    width=600,
    height=600
)

# Agregar total en el centro
total_days = cause_summary["duration_days"].sum()
fig_donut.update_layout(
    annotations=[dict(text=f"{total_days} days", x=0.5, y=0.5, font_size=20, showarrow=False)]
)

st.plotly_chart(fig_donut, use_container_width=True)

#ESG Graph

#ESG Keywords for voluntary experience classification
st.header("Volunteering Experience in ESG Areas")

keywords_esg = {
    "Environmental": [
        "tree planting", "climate", "recycling", "sustainability", 
        "biodiversity", "conservation", "renewable", "waste","reforestation","reforestation and tree planting","ecosystem restoration",
        "beach clean up","river clean up","endangered species protection","biodiversity conservation","community garden","wetland restoration",
        "marine coral protection","marine coral restoration","wildlife monitoring","biological corridor creation","solar panel installation",
        "clean energy promotion","rainwater harvesting system construction","green technology implementation","home energy audits","sustainable transportation",
        "urban garden development","community composting","organic farming ","wind energy system","community recycling programs","plastic waste reduction",
        "waste separation campaigns","material reuse","upcycling workshops","electronic waste management","organic waste programs","clean point creation",
        "responsible consumption education","used item exchange","ecological awareness","climate change educational programs","permaculture training",
        "clean water education","environmental awareness","sustainable agriculture training","green building","pollution education","responsible ecotourism ",
        "renewable energy","sustainability education","solidarity economy","fair trade"
    ],
    "Social": [
        "education", "health", "volunteering", "community","adult literacy","gender", "equity", "training", "child", "youth","academic tutoring",
        "children's reading programs","digital education","language classes for immigrants","professional training","financial education","student scholarship programs",
        "university mentoring","vaccination campaigns","mental health programs","medical care in rural communities","rural communities","disease prevention campaigns",
        "blood donation","elderly care","child nutrition programs","recreational therapies","community first aid","homeless assistance","domestic violence victim support",
        "Programs for people with disabilities","disabilities","refugee and immigrant assistance", "migrant assistance", "refugees","migrants","orphan care","single-parent family support",
        "indigenous community assistance","war veteran support","ex-prisoner programs","gender equality","racial discrimination","LGBTI+","disability inclusion","women's empowerment",
        "anti-bullying ","cultural diversity","children's rights","social justice","equal access to opportunities","social housing construction","community infrastructure development",
        "microcredit programs","local organization strengthening","cooperative development","social entrepreneurship programs","neighborhood improvement","recreational space creation",
        "rural development","food security programs","community art workshops","cultural heritage preservation","youth music programs","social theater","community cultural festivals",
        "community libraries","local tradition promotion","art therapy","community cinema","adaptive sports","natural disaster response","humanitarian aid distribution","temporary shelters",
        "search and rescue","post-disaster rehabilitation","risk prevention","emergency first aid","community evacuations","crisis psychological support","community reconstruction",
        "sustainability education","solidarity economy","fair trade"
      
    
    ],
    "Governance": [
        "ethics", "transparency", "compliance", "anti-corruption","citizen audits","public works monitoring","citizen oversight",
        "public information access","accountability", "governance", "board", "policy","transparency reports","social resource control",
        "political promise tracking","public program evaluation","corruption reporting","development council participation","civic education",
        "conscious voting promotion","community leader training","popular consultation participation","citizen assemblies","open town halls","participatory budgets",
        "community councils","public forums","citizen dialogue tables","anti-corruption programs","civic values promotion","community ethics codes","integrity training",
        "ethics committees","conflict mediation","restorative justice","values programs","business ethics","corporate social responsibility","public official training",
        "government process improvement","civil organization strengthening","institutional capacity development","quality management systems","technological modernization",
        "administrative efficiency","results-based management","strategic planning","public policy evaluation","public policy design","citizen legislative proposals","citizen proposals",
        "policy impact analysis","implementation monitoring","social program evaluation","public policy research","advocacy networks","citizen coalitions","popular legal education",
        "free legal advice","constitutional rights defense","access to justice","legal reforms","strategic litigation","electoral observation","human rights","social justice",
        "electronic government","digital participation","technological transparency","citizen digital literacy","open data","online participation platforms",
        "citizen cybersecurity","data privacy","digital inclusion","government innovation","participatory urban planning","territorial planning","local sustainable development",
        "community development plans","participatory zoning","territory management","local strategic planning","environmental territorial management","Sustainability education",
        "solidarity economy","fair trade"
    ]
}

model_2 = SentenceTransformer("distiluse-base-multilingual-cased-v2")

esg_labels = ["Environment", "Social", "Governance"]

label_embeddings = model_2.encode(esg_labels, convert_to_tensor=True)

def esg_scoring(text, model=model_2, label_embeddings=label_embeddings, cleaning_fn=normalize_text):
    
    #if the description column is empty, we assign parity
    if not isinstance(text, str) or text.strip() == "":
     scores = np.ones(len(esg_labels)) / len(esg_labels)
    else:

        cleaned_text = cleaning_fn(text)
        text_emb = model.encode(cleaned_text, convert_to_tensor=True)

        cos_sim = util.cos_sim(text_emb, label_embeddings).cpu().numpy().flatten()
    if cos_sim.sum() == 0:
            scores = np.ones_like(cos_sim) / len(cos_sim)
    else:
            scores = cos_sim / cos_sim.sum()

    return dict(zip(esg_labels, scores))
#Apply ESG scoring to the volunteering descriptions
esg_results = vol_candidate['description'].apply(esg_scoring)
esg_df = pd.DataFrame(list(esg_results))
esg_df['candidate_id'] = vol_candidate['candidate_id'].values
#Final ESG dataframe with candidate_id and the ESG scores
esg_df = esg_df[['candidate_id'] + esg_labels]

total_scores = esg_df[esg_labels].sum()
if total_scores.sum() == 0:
    st.warning("No ESG-related volunteering experience found for the selected candidate.")
#Convert to percentages
percentages_esg= (total_scores / total_scores.sum() * 100).round(2)

fig, ax = plt.subplots(figsize=(6, 3))
percentages_esg.plot(kind='bar', color=custom_palette, ax=ax)
ax.set_ylim(0, 100)
ax.set_ylabel('Percentage (%)')
#orientation of x labels
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='right')
#No frame
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.tight_layout()
plt.close()
st.pyplot(fig)

#Create international profile

st.header("International Profile")

#Country code dictionary
country_codes = {
    # A
    'afghanistan': 'AFG', 'afganistan': 'AFG', 'afghan': 'AFG',
    'albania': 'ALB', 'albanian': 'ALB',
    'algeria': 'DZA', 'argelia': 'DZA', 'algerian': 'DZA',
    'andorra': 'AND', 'andorran': 'AND',
    'angola': 'AGO', 'angolan': 'AGO',
    'antigua and barbuda': 'ATG', 'antigua y barbuda': 'ATG',
    'argentina': 'ARG', 'argentinean': 'ARG', 'argentinian': 'ARG', 'argentino': 'ARG',
    'armenia': 'ARM', 'armenian': 'ARM',
    'australia': 'AUS', 'australian': 'AUS',
    'austria': 'AUT', 'austrian': 'AUT',
    'azerbaijan': 'AZE', 'azerbaiyan': 'AZE', 'azerbaijani': 'AZE',
    
    # B
    'bahamas': 'BHS', 'bahamian': 'BHS',
    'bahrain': 'BHR', 'bahrein': 'BHR', 'bahraini': 'BHR',
    'bangladesh': 'BGD', 'bangladeshi': 'BGD',
    'barbados': 'BRB', 'barbadian': 'BRB',
    'belarus': 'BLR', 'bielorrusia': 'BLR', 'belarusian': 'BLR',
    'belgium': 'BEL', 'belgica': 'BEL', 'belgian': 'BEL',
    'belize': 'BLZ', 'belizean': 'BLZ',
    'benin': 'BEN', 'beninese': 'BEN',
    'bhutan': 'BTN', 'butan': 'BTN', 'bhutanese': 'BTN',
    'bolivia': 'BOL', 'bolivian': 'BOL', 'boliviano': 'BOL',
    'bosnia and herzegovina': 'BIH', 'bosnia y herzegovina': 'BIH', 'bosnian': 'BIH',
    'botswana': 'BWA', 'botswanan': 'BWA',
    'brazil': 'BRA', 'brasil': 'BRA', 'brazilian': 'BRA', 'brasileno': 'BRA',
    'brunei': 'BRN', 'bruneian': 'BRN',
    'bulgaria': 'BGR', 'bulgarian': 'BGR',
    'burkina faso': 'BFA', 'burkinabe': 'BFA',
    'burundi': 'BDI', 'burundian': 'BDI',
    
    # C
    'cambodia': 'KHM', 'camboya': 'KHM', 'cambodian': 'KHM',
    'cameroon': 'CMR', 'camerun': 'CMR', 'cameroonian': 'CMR',
    'canada': 'CAN', 'canadian': 'CAN', 'canadiense': 'CAN',
    'cape verde': 'CPV', 'cabo verde': 'CPV', 'cape verdean': 'CPV',
    'central african republic': 'CAF', 'republica centroafricana': 'CAF',
    'chad': 'TCD', 'chadian': 'TCD',
    'chile': 'CHL', 'chilean': 'CHL', 'chileno': 'CHL',
    'china': 'CHN', 'chinese': 'CHN', 'chino': 'CHN',
    'colombia': 'COL', 'colombian': 'COL', 'colombiano': 'COL',
    'comoros': 'COM', 'comoras': 'COM', 'comorian': 'COM',
    'congo': 'COG', 'republic of congo': 'COG', 'congolese': 'COG',
    'democratic republic of congo': 'COD', 'congo democratic republic': 'COD',
    'costa rica': 'CRI', 'costa rican': 'CRI', 'costarricense': 'CRI',
    'croatia': 'HRV', 'croacia': 'HRV', 'croatian': 'HRV',
    'cuba': 'CUB', 'cuban': 'CUB', 'cubano': 'CUB',
    'cyprus': 'CYP', 'chipre': 'CYP', 'cypriot': 'CYP',
    'czech republic': 'CZE', 'republica checa': 'CZE', 'czech': 'CZE',
    
    # D
    'denmark': 'DNK', 'dinamarca': 'DNK', 'danish': 'DNK', 'danes': 'DNK',
    'djibouti': 'DJI', 'djiboutian': 'DJI',
    'dominica': 'DMA', 'dominican commonwealth': 'DMA',
    'dominican republic': 'DOM', 'republica dominicana': 'DOM', 'dominican': 'DOM',
    
    # E
    'ecuador': 'ECU', 'ecuadorian': 'ECU', 'ecuatoriano': 'ECU',
    'egypt': 'EGY', 'egipto': 'EGY', 'egyptian': 'EGY',
    'el salvador': 'SLV', 'salvador': 'SLV', 'salvadoran': 'SLV', 'salvadoreno': 'SLV',
    'equatorial guinea': 'GNQ', 'guinea ecuatorial': 'GNQ',
    'eritrea': 'ERI', 'eritrean': 'ERI',
    'estonia': 'EST', 'estonian': 'EST',
    'eswatini': 'SWZ', 'swaziland': 'SWZ', 'swazi': 'SWZ',
    'ethiopia': 'ETH', 'etiopia': 'ETH', 'ethiopian': 'ETH',
    
    # F
    'fiji': 'FJI', 'fijian': 'FJI',
    'finland': 'FIN', 'finlandia': 'FIN', 'finnish': 'FIN', 'finn': 'FIN',
    'france': 'FRA', 'francia': 'FRA', 'french': 'FRA', 'frances': 'FRA',
    
    # G
    'gabon': 'GAB', 'gabonese': 'GAB',
    'gambia': 'GMB', 'gambian': 'GMB',
    'georgia': 'GEO', 'georgian': 'GEO',
    'germany': 'DEU', 'alemania': 'DEU', 'german': 'DEU', 'aleman': 'DEU',
    'ghana': 'GHA', 'ghanaian': 'GHA',
    'greece': 'GRC', 'grecia': 'GRC', 'greek': 'GRC', 'griego': 'GRC',
    'grenada': 'GRD', 'grenadian': 'GRD',
    'guatemala': 'GTM', 'guatemalan': 'GTM', 'guatemalteco': 'GTM',
    'guinea': 'GIN', 'guinean': 'GIN',
    'guinea-bissau': 'GNB', 'guinea bissau': 'GNB',
    'guyana': 'GUY', 'guyanese': 'GUY',
    
    # H
    'haiti': 'HTI', 'haitian': 'HTI', 'haitiano': 'HTI',
    'honduras': 'HND', 'honduran': 'HND', 'hondureno': 'HND',
    'hungary': 'HUN', 'hungria': 'HUN', 'hungarian': 'HUN',
    
    # I
    'iceland': 'ISL', 'islandia': 'ISL', 'icelandic': 'ISL',
    'india': 'IND', 'indian': 'IND', 'indio': 'IND',
    'indonesia': 'IDN', 'indonesian': 'IDN', 'indonesio': 'IDN',
    'iran': 'IRN', 'iranian': 'IRN', 'irani': 'IRN', 'persian': 'IRN',
    'iraq': 'IRQ', 'iraqi': 'IRQ',
    'ireland': 'IRL', 'irlanda': 'IRL', 'irish': 'IRL', 'irlandes': 'IRL',
    'israel': 'ISR', 'israeli': 'ISR',
    'italy': 'ITA', 'italia': 'ITA', 'italian': 'ITA', 'italiano': 'ITA',
    'ivory coast': 'CIV', 'cote divoire': 'CIV', 'costa de marfil': 'CIV',
    
    # J
    'jamaica': 'JAM', 'jamaican': 'JAM', 'jamaiquino': 'JAM',
    'japan': 'JPN', 'japon': 'JPN', 'japanese': 'JPN', 'japones': 'JPN',
    'jordan': 'JOR', 'jordania': 'JOR', 'jordanian': 'JOR',
    
    # K
    'kazakhstan': 'KAZ', 'kazajistan': 'KAZ', 'kazakh': 'KAZ',
    'kenya': 'KEN', 'kenyan': 'KEN',
    'kiribati': 'KIR', 'kiribatian': 'KIR',
    'north korea': 'PRK', 'corea del norte': 'PRK', 'north korean': 'PRK',
    'south korea': 'KOR', 'corea del sur': 'KOR', 'korean': 'KOR', 'coreano': 'KOR',
    'kuwait': 'KWT', 'kuwaiti': 'KWT',
    'kyrgyzstan': 'KGZ', 'kirguistan': 'KGZ', 'kyrgyz': 'KGZ',
    
    # L
    'laos': 'LAO', 'lao': 'LAO', 'laotian': 'LAO',
    'latvia': 'LVA', 'letonia': 'LVA', 'latvian': 'LVA',
    'lebanon': 'LBN', 'libano': 'LBN', 'lebanese': 'LBN',
    'lesotho': 'LSO', 'lesothan': 'LSO',
    'liberia': 'LBR', 'liberian': 'LBR',
    'libya': 'LBY', 'libia': 'LBY', 'libyan': 'LBY',
    'liechtenstein': 'LIE', 'liechtensteiner': 'LIE',
    'lithuania': 'LTU', 'lituania': 'LTU', 'lithuanian': 'LTU',
    'luxembourg': 'LUX', 'luxemburgo': 'LUX', 'luxembourger': 'LUX',
    
    # M
    'madagascar': 'MDG', 'malagasy': 'MDG',
    'malawi': 'MWI', 'malawian': 'MWI',
    'malaysia': 'MYS', 'malasia': 'MYS', 'malaysian': 'MYS',
    'maldives': 'MDV', 'maldivas': 'MDV', 'maldivian': 'MDV',
    'mali': 'MLI', 'malian': 'MLI',
    'malta': 'MLT', 'maltese': 'MLT', 'maltes': 'MLT',
    'marshall islands': 'MHL', 'islas marshall': 'MHL', 'marshallese': 'MHL',
    'mauritania': 'MRT', 'mauritanian': 'MRT',
    'mauritius': 'MUS', 'mauricio': 'MUS', 'mauritian': 'MUS',
    'mexico': 'MEX', 'mexican': 'MEX', 'mexicano': 'MEX',
    'micronesia': 'FSM', 'micronesian': 'FSM',
    'moldova': 'MDA', 'moldovan': 'MDA',
    'monaco': 'MCO', 'monacan': 'MCO', 'monegasque': 'MCO',
    'mongolia': 'MNG', 'mongolian': 'MNG', 'mongol': 'MNG',
    'montenegro': 'MNE', 'montenegrin': 'MNE',
    'morocco': 'MAR', 'marruecos': 'MAR', 'moroccan': 'MAR',
    'mozambique': 'MOZ', 'mozambican': 'MOZ',
    'myanmar': 'MMR', 'burma': 'MMR', 'burmese': 'MMR',
    
    # N
    'namibia': 'NAM', 'namibian': 'NAM',
    'nauru': 'NRU', 'nauruan': 'NRU',
    'nepal': 'NPL', 'nepalese': 'NPL', 'nepali': 'NPL',
    'netherlands': 'NLD', 'paises bajos': 'NLD', 'dutch': 'NLD', 'holandes': 'NLD',
    'new zealand': 'NZL', 'nueva zelanda': 'NZL', 'new zealander': 'NZL',
    'nicaragua': 'NIC', 'nicaraguan': 'NIC', 'nicaraguense': 'NIC',
    'niger': 'NER', 'nigerien': 'NER',
    'nigeria': 'NGA', 'nigerian': 'NGA',
    'north macedonia': 'MKD', 'macedonia del norte': 'MKD', 'macedonian': 'MKD',
    'norway': 'NOR', 'noruega': 'NOR', 'norwegian': 'NOR', 'noruego': 'NOR',
    
    # O
    'oman': 'OMN', 'omani': 'OMN',
    
    # P
    'pakistan': 'PAK', 'pakistani': 'PAK',
    'palau': 'PLW', 'palauan': 'PLW',
    'palestine': 'PSE', 'palestina': 'PSE', 'palestinian': 'PSE',
    'panama': 'PAN', 'panamanian': 'PAN', 'panameno': 'PAN',
    'papua new guinea': 'PNG', 'papua nueva guinea': 'PNG', 'papua new guinean': 'PNG',
    'paraguay': 'PRY', 'paraguayan': 'PRY', 'paraguayo': 'PRY',
    'peru': 'PER', 'peruvian': 'PER', 'peruano': 'PER',
    'philippines': 'PHL', 'filipinas': 'PHL', 'filipino': 'PHL',
    'poland': 'POL', 'polonia': 'POL', 'polish': 'POL', 'polaco': 'POL',
    'portugal': 'PRT', 'portuguese': 'PRT', 'portugues': 'PRT',
    
    # Q
    'qatar': 'QAT', 'qatari': 'QAT',
    
    # R
    'romania': 'ROU', 'rumania': 'ROU', 'romanian': 'ROU', 'rumano': 'ROU',
    'russia': 'RUS', 'rusia': 'RUS', 'russian': 'RUS', 'ruso': 'RUS',
    'rwanda': 'RWA', 'rwandan': 'RWA',
    
    # S
    'saint kitts and nevis': 'KNA', 'san cristobal y nieves': 'KNA',
    'saint lucia': 'LCA', 'santa lucia': 'LCA', 'saint lucian': 'LCA',
    'saint vincent and the grenadines': 'VCT', 'san vicente y las granadinas': 'VCT',
    'samoa': 'WSM', 'samoan': 'WSM',
    'san marino': 'SMR', 'sanmarinese': 'SMR',
    'sao tome and principe': 'STP', 'santo tome y principe': 'STP',
    'saudi arabia': 'SAU', 'arabia saudita': 'SAU', 'saudi': 'SAU', 'arab': 'SAU',
    'senegal': 'SEN', 'senegalese': 'SEN',
    'serbia': 'SRB', 'serbian': 'SRB',
    'seychelles': 'SYC', 'seychellois': 'SYC',
    'sierra leone': 'SLE', 'sierra leonean': 'SLE',
    'singapore': 'SGP', 'singapur': 'SGP', 'singaporean': 'SGP',
    'slovakia': 'SVK', 'eslovaquia': 'SVK', 'slovak': 'SVK',
    'slovenia': 'SVN', 'eslovenia': 'SVN', 'slovenian': 'SVN',
    'solomon islands': 'SLB', 'islas salomon': 'SLB', 'solomon islander': 'SLB',
    'somalia': 'SOM', 'somali': 'SOM',
    'south africa': 'ZAF', 'sudafrica': 'ZAF', 'south african': 'ZAF',
    'south sudan': 'SSD', 'sudan del sur': 'SSD', 'south sudanese': 'SSD',
    'spain': 'ESP', 'espana': 'ESP', 'spanish': 'ESP', 'espanol': 'ESP',
    'sri lanka': 'LKA', 'sri lankan': 'LKA',
    'sudan': 'SDN', 'sudanese': 'SDN',
    'suriname': 'SUR', 'surinamese': 'SUR',
    'sweden': 'SWE', 'suecia': 'SWE', 'swedish': 'SWE', 'sueco': 'SWE',
    'switzerland': 'CHE', 'suiza': 'CHE', 'swiss': 'CHE', 'suizo': 'CHE',
    'syria': 'SYR', 'siria': 'SYR', 'syrian': 'SYR',
    
    # T
    'tajikistan': 'TJK', 'tayikistan': 'TJK', 'tajik': 'TJK',
    'tanzania': 'TZA', 'tanzanian': 'TZA',
    'thailand': 'THA', 'tailandia': 'THA', 'thai': 'THA',
    'timor-leste': 'TLS', 'east timor': 'TLS', 'timorese': 'TLS',
    'togo': 'TGO', 'togolese': 'TGO',
    'tonga': 'TON', 'tongan': 'TON',
    'trinidad and tobago': 'TTO', 'trinidad y tobago': 'TTO', 'trinidadian': 'TTO',
    'tunisia': 'TUN', 'tunez': 'TUN', 'tunisian': 'TUN',
    'turkey': 'TUR', 'turquia': 'TUR', 'turkish': 'TUR', 'turco': 'TUR',
    'turkmenistan': 'TKM', 'turkmen': 'TKM',
    'tuvalu': 'TUV', 'tuvaluan': 'TUV',
    
    # U
    'uganda': 'UGA', 'ugandan': 'UGA',
    'ukraine': 'UKR', 'ucrania': 'UKR', 'ukrainian': 'UKR', 'ucraniano': 'UKR',
    'united arab emirates': 'ARE', 'emiratos arabes unidos': 'ARE', 'emirati': 'ARE',
    'united kingdom': 'GBR', 'reino unido': 'GBR', 'british': 'GBR', 'britanico': 'GBR',
    'uk': 'GBR', 'great britain': 'GBR', 'england': 'GBR', 'english': 'GBR',
    'scotland': 'GBR', 'scottish': 'GBR', 'wales': 'GBR', 'welsh': 'GBR',
    'northern ireland': 'GBR', 'irish': 'IRL',
    'united states': 'USA', 'estados unidos': 'USA', 'american': 'USA', 'estadounidense': 'USA',
    'usa': 'USA', 'us': 'USA', 'america': 'USA',
    'uruguay': 'URY', 'uruguayan': 'URY', 'uruguayo': 'URY',
    'uzbekistan': 'UZB', 'uzbek': 'UZB',
    
    # V
    'vanuatu': 'VUT', 'vanuatuan': 'VUT',
    'vatican city': 'VAT', 'ciudad del vaticano': 'VAT', 'vatican': 'VAT',
    'venezuela': 'VEN', 'venezuelan': 'VEN', 'venezolano': 'VEN',
    'vietnam': 'VNM', 'vietnamese': 'VNM', 'vietnamita': 'VNM',
    
    # Y
    'yemen': 'YEM', 'yemeni': 'YEM',
    
    # Z
    'zambia': 'ZMB', 'zambian': 'ZMB',
    'zimbabwe': 'ZWE', 'zimbabwean': 'ZWE'
}
#Function to standarize time abroad
def standardize_time_abroad(entry):
#transforming from the entry format 1y, 6m to total months
    if pd.isna(entry) or entry == '':
        return 0
    entry = str(entry).lower().strip() #lowercase and strip spaces
    total_months = 0
#years
    years_match = re.search(r'(\d+)y', entry)
    if years_match:
        total_months += int(years_match.group(1)) * 12
    #months
    months_match = re.search(r'(\d+)m', entry)
    if months_match:

        total_months += int(months_match.group(1))
    
    return total_months

#Function to classify cultural heritage
def parse_cultural_heritage(heritage_str):
    """Parse cultural heritage like 'mexican-chinese' or 'arab' into list"""
    if pd.isna(heritage_str) or heritage_str == '':
        return []
    
    # Split by dash, comma, or space
    heritages = re.split(r'[-,\s]+', str(heritage_str).lower())
    return [h.strip() for h in heritages if h.strip()]





# Filter the international experience dataframe for the selected candidate

int_candidate = int_df[int_df['candidate_id'] == selected_id].copy()

# Create comprehensive country data list
all_countries = []

if len(int_candidate) > 0:
    # Get candidate info (assuming same for all rows)
    candidate_info = int_candidate.iloc[0]
    
    # 1. INTERNATIONAL EXPERIENCES
    int_candidate['time_months'] = int_candidate['time_abroad'].apply(standardize_time_abroad)
    
    for _, row in int_candidate.iterrows():
        if pd.notna(row['country_of_experience']) and row['country_of_experience'] != '':
            all_countries.append({
                'country': row['country_of_experience'],
                'time_months': row['time_months'],
                'source': 'International Experience'
            })
    
    # 2. COUNTRY OF ORIGIN
    if pd.notna(candidate_info['country_of_origin']) and candidate_info['country_of_origin'] != '':
        all_countries.append({
            'country': candidate_info['country_of_origin'],
            'time_months': 240,  # 20 years default
            'source': 'Country of Origin'
        })
    
    # 3. COUNTRY OF RESIDENCY
    if pd.notna(candidate_info['country_of_residence']) and candidate_info['country_of_residence'] != '':
        if candidate_info['country_of_residence'] != candidate_info['country_of_origin']:
            all_countries.append({
                'country': candidate_info['country_of_residence'],
                'time_months': 120,  # 10 years default
                'source': 'Country of Residency'
            })
    
    # 4. CULTURAL HERITAGE
    if pd.notna(candidate_info['cultural_heritage']) and candidate_info['cultural_heritage'] != '':
        heritage_list = parse_cultural_heritage(candidate_info['cultural_heritage'])
        
        heritage_mapping = {
            'mexican': 'mexico', 'chinese': 'china', 'arab': 'saudi arabia',
            'german': 'germany', 'italian': 'italy', 'french': 'france',
            'spanish': 'spain', 'japanese': 'japan', 'korean': 'south korea',
            'indian': 'india', 'brazilian': 'brazil', 'russian': 'russia'
        }
        
        for heritage in heritage_list:
            heritage_country = heritage_mapping.get(heritage, heritage)
            all_countries.append({
                'country': heritage_country,
                'time_months': 60,  # 5 years default
                'source': 'Cultural Heritage'
            })
    
    # Convert to DataFrame and process
    if all_countries:
        countries_df = pd.DataFrame(all_countries)
        
        # Group by country and sum time
        map_data = countries_df.groupby('country').agg({
            'time_months': 'sum',
            'source': lambda x: ', '.join(x.unique())
        }).reset_index()
        
        # Convert to years
        map_data['time_years'] = (map_data['time_months'] / 12).round(2)
        
        # Map to ISO codes
        map_data['country_normalized'] = map_data['country'].str.lower().str.strip()
        map_data['iso_code'] = map_data['country_normalized'].map(country_codes)
        
        # Keep only valid countries
        map_data = map_data.dropna(subset=['iso_code'])

       




# Check if we have valid data
if not map_data.empty:
    try:
        # Create the map
        fig = px.choropleth(
            map_data,
            locations='iso_code',
            color='time_years',
            hover_name='country',
            hover_data={
                'source': True,
                'time_months': ':.0f',
                'time_years': ':.1f',
                'iso_code': False
            },
            color_continuous_scale='Sunset',
            title='Multidimensional Cultural Map',
            labels={
                'time_years': 'Years of Connection',
                'time_months': 'Months',
                'source': 'Connection Type'
            }
        )

        fig.update_layout(
            geo=dict(
                showframe=False,
                showcoastlines=True,
                showland=True,
                landcolor='rgb(243, 243, 243)',
                coastlinecolor='rgb(204, 204, 204)',
            ),
            coloraxis_colorbar=dict(
                title="Years of Connection"
            ),
            height=600
        )

        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating map: {str(e)}")
        
else:
    st.warning(f"No international experience data found for the selected candidate.")