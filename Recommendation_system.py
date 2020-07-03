
# coding: utf-8

# # Netflix is getting every much popular in our nation with its content , shows and movies. This is an EDA through its data and content based recommendations system

# In[1]:


#importing required libraries


# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# # loading Dataset

# In[2]:


netflix_data = pd.read_csv('netflix_titles.csv')


# In[3]:


# first 10 entries of the data
netflix_data.head(10)


# In[4]:


# so there are 12 columns in the dataset


# In[5]:


netflix_data.count()


# In[6]:


# let's see about total number of TV shows
netflix_Tv_shows = netflix_data[netflix_data['type'] == 'TV Show']
#print(netflix_Tv_shows.count()) #1969


# In[7]:


# Total number of movies
netflix_movies = netflix_data[netflix_data['type']=='Movie']
#print(netflix_movies.count())    #4265


# # Visualize Movie vs TV shows 

# In[8]:


sns.set(style='whitegrid')
sns.countplot(x='type',data = netflix_data,palette="Set3") # countplot Shows the count of observation in each categorical bins 


# Its shows that there are more movies on netflix than Tv shows

# # If a production house wants to release some content which month that they do so ?

# In[9]:


netflix_date = netflix_data[['date_added']].dropna()
# here  we have to date like september 9 ,2019 we want to extract year and month from the date_added
netflix_date['year'] = netflix_date['date_added'].apply(lambda x : x.split(', ')[-1]) # 2019
netflix_date['month'] = netflix_date['date_added'].apply(lambda x: x.lstrip().split(' ')[0]) # september
months_order = ['January','February','March','April','May','June','July','August','September','October','November','December'][::-1] # reversed the list
df = netflix_date.groupby('year')['month'].value_counts().unstack().fillna(0).T
plt.figure(figsize=(10,7) ,dpi= 200)
plt.pcolor(df,cmap='afmhot_r',edgecolors = 'black',linewidths = 3) #heatmap
plt.xticks(np.arange(0.5,len(df.columns),1),df.columns,fontsize=7)
plt.yticks(np.arange(0.5, len(df.index), 1), df.index, fontsize=7)
plt.title('Netflix Contents Update', fontsize=12, fontweight='bold', position=(0.20, 1.0+0.02))
cbar = plt.colorbar()
# cbar.ax.tick_params(labelsize=8) 
# cbar.ax.minorticks_on()
plt.show()


# # Tv show rating analysis
# 
# 

# In[10]:


plt.figure(figsize=(10,7))
sns.set(style='darkgrid')
sns.countplot(x='rating',data = netflix_Tv_shows,palette = "Set1",order = netflix_Tv_shows['rating'].value_counts().index[0:15])


# # Analysing IMDB ratings to get top rated movies on Netflix

# In[11]:


# loading dataset of IMDB ratings of movies
imdb_ratings = pd.read_csv('IMDB ratings.csv',usecols=['weighted_average_vote'])
imdb_title = pd.read_csv('IMDB movies.csv',usecols=['title','year','genre'])
# So now new data frame to create that is title , year , genre and rating (weighted_average_vote)
ratings = pd.DataFrame({'Title':imdb_title.title,
                       'Release Year':imdb_title.year,
                       'Genre':imdb_title.genre,
                       'Ratings':imdb_ratings.weighted_average_vote})


# # Performing inner join on ratings dataset and netflix dataset to get the content that has both  ratings on IMDB and are available in Netflix

# In[12]:


ratings.dropna()
join_data = ratings.merge(netflix_data,left_on='Title',right_on='title',how='inner')
join_data = join_data.sort_values(by='Ratings',ascending=False)


# In[13]:


# Top 10 rated movies on net
top_rated = join_data[0:10]
plt.figure(figsize = (25,8))
sns.set(style="whitegrid",font = 'Calibri')
sns.barplot(x = "Title",y = "Ratings" ,data = top_rated)


# # Year wise analysis

# In[14]:


plt.figure(figsize = (12,10))
sns.set(style = 'whitegrid')
ax = sns.countplot(y = "release_year",data = netflix_movies,palette = "Set1",order = netflix_movies["release_year"].value_counts().index[0:10])


# So 2017 was the year when more content has beeen released 

# # Analysis of Duration of movies

# In[20]:



sns.set(style = 'whitegrid')
sns.kdeplot(data = netflix_movies['duration'],shade=True)


#  So, a good amount of movies on Netflix are among the duration of 75-120 mins.

# 
# # WordCloud for Genres

# In[21]:


from collections import Counter

genres = list(netflix_movies['listed_in'])
gen = []
for i in genres:
    i = list(i.split(','))
    for j in i:
        gen.append(j.replace(" ",''))
g = Counter(gen)


# In[22]:


# Different count of the genre
plt.figure(figsize=(50,15))
labels , values = zip(*g.items())
indexes = np.arange(len(labels))
plt.bar(indexes,values)
plt.xticks(indexes,labels)
plt.show


# In[23]:


from wordcloud import WordCloud , STOPWORDS , ImageColorGenerator
text = list(set(gen)) # unique genre
plt.rcParams['figure.figsize'] = (13,13)
wordcloud = WordCloud(max_font_size=50,max_words=100,background_color='white').generate(str(text))
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis('off')
plt.show()


# # Countries with highest rated content

# In[26]:


country_count = join_data['country'].value_counts().sort_values(ascending=False)
country_count = pd.DataFrame(country_count)
top_country = country_count[0:10]
top_country


# In[27]:


# Funnel chart using plotly

import plotly.express as px
data = dict(number=[1063,619,135,60,47,44,41,40,40,38],
           country=["United States","India","United Kingdom","Canada","UK,US","Spain",
                    "Turkey","Philippines","France","South Korea"])
fig = px.funnel(data,x='number',y='country')
fig.show()


# # TV shows with Largest Number of Seasons

# In[39]:


features = ['title','duration']
durations = netflix_Tv_shows[features]
durations['no_of_season'] = durations['duration'].str.replace('Season','')
durations['no_of_season'] = durations['no_of_season'].str.replace('s','')
durations['no_of_season'] = durations['no_of_season'].astype(str).astype(int)
top = ['title','no_of_season']
top_ = durations[top]
top_ = top_.sort_values(by = 'no_of_season',ascending = False)
top20=top_[0:20]
top20.plot(kind = 'bar',x='title',y='no_of_season',color='pink')


# Thus, NCIS, Grey's Anatomy and Supernatural are amongst the tv series that have highest number of seasons.

# # Oldest and newest Tv shows by Country

# In[55]:


country_series_data = netflix_Tv_shows[netflix_Tv_shows['country']== str(input('Enter Your Country: '))]


# In[56]:


oldest_series = country_series_data.sort_values(by='release_year')[0:20]
import plotly.graph_objects as go
fig = go.Figure(data=[go.Table(header = dict(values = ['Title' ,'Release Year'],fill_color='paleturquoise'),
                               cells = dict(values=[oldest_series['title'],oldest_series['release_year']],fill_color='orange'))])
fig.show()


# Above Table shows oldest shows by country

# In[59]:


# newest tv shows by country
newest_series = country_series_data.sort_values(by='release_year',ascending=False)[0:25]
fig = go.Figure(data=[go.Table(header = dict(values=['Title','Release Year'],fill_color='paleturquoise'),
                              cells = dict(values=[newest_series['title'],newest_series['release_year']],fill_color='lavender'))])
fig.show()


# Above Table shows newest shows by country

# # Now comes the most important part Recommendation System (Content Based)

# In[61]:


# The TF-IDF (Term frequency - Inverse document frequency) score is the frequency of a word occuring in a document , 
#down weighted by the number of documents in which it occurs


# In[67]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english')  # removing stopwords
netflix_data['description'] = netflix_data['description'].fillna('') # Replacing NAN with an empty string
tfidf_matrix = tfidf.fit_transform(netflix_data['description'])
tfidf_matrix.shape


# There are about 16151 words described for the 6234 movies in this dataset

# Here, The Cosine similarity score is used since it is independent of magnitude and is relatively easy and fast to calculate.

# In[68]:


# import cosine_similarity 
from sklearn.metrics.pairwise import cosine_similarity
cosine_sim = cosine_similarity(tfidf_matrix,tfidf_matrix)


# In[69]:


cosine_sim


# In[75]:


indices = pd.Series(netflix_data.index,index = netflix_data['title']).drop_duplicates()


# In[76]:


# Recommendation Function

def get_recommendation(title,cosine_sim=cosine_sim):
    idx = indices[title]
    
    sim_scores = list(enumerate(cosine_sim[idx])) # Get the pairwsie similarity scores of all movies with that movie
    
    sim_scores = sorted(sim_scores,key = lambda x :x[1],reverse = True) # Sort the movies based on the similarity scores
    
    sim_scores = sim_scores[1:11]  # Get the scores of the 10 most similar movies
    
    movie_indices = [i[0] for i in sim_scores] # Get the movie indices
    
    
    return netflix_data['title'].iloc[movie_indices]


# In[77]:


get_recommendation('Peaky Blinders')


# The above recommendations is based on plot

# # let's add more metrices to improve the model performance

# Content based filtering on following factors
# 1) Title
# 2) Cast
# 3) Director
# 4) listed_in

# In[78]:


# Fill null value with empty string

filledna = netflix_data.fillna('')


# In[79]:


# Clean the data , making all the words in lower case

def clean_data (x):
    return str.lower(x.replace(" ",''))


# In[80]:


features=['title','director','cast','listed_in','description'] # identifying features on which model is to be filtered
filledna=filledna[features]


# In[83]:


for feature in features:
    filledna[feature] = filledna[feature].apply(clean_data)
filledna.head(5)


# In[84]:


# Create a bag of words for all rows 
def bag_words(x):
    return x['title']+' '+x['director']+' '+x['cast']+' '+x['listed_in']+' '+x['description']


# In[85]:


filledna['bag_words'] = filledna.apply(bag_words,axis=1)


# Now instead of using TF-IDF were going to use CountVectorizer

# In[88]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

count =CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(filledna['bag_words'])
cosine_sim2 = cosine_similarity(count_matrix, count_matrix)


# In[89]:


filledna=filledna.reset_index()
indices = pd.Series(filledna.index, index=filledna['title'])


# In[92]:


def get_recommendations_new(title, cosine_sim=cosine_sim):
    title=title.replace(' ','').lower()
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return netflix_data['title'].iloc[movie_indices]


# In[94]:


get_recommendations_new('Peaky Blinders',cosine_sim2)


# In[100]:


get_recommendations_new('You',cosine_sim2)


# # So this was it ! Hope so you liked it 
