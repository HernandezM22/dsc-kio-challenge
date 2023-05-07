import streamlit as st

st.set_page_config(
    page_title="Server Anomaly Detection",
    layout="wide",
    menu_items={
        'about': "Mauricio Hernandez, Jaime Guzman, Brenda Martinez, Ricardo Diaz"
    }
)

st.title("Welcome to the documentation of the anomaly detection tool")
st.markdown('## Understanding The Data')
st.markdown('### Parse')
st.markdown('The format of the dataset is given in a csv file where inside the file contains a column with a json format variable. ')
st.markdown('Which we need to convert(parse) the json file to strings in order that we can make an EDA(Eploratory Data Analysis).')
st.markdown('The data was transformed from .csv to .pkl for compression and also for faster read/write (mostly read). The data can be considered')
st.markdown('to be separated in three sections, the network, memory and cpu section. We divided these individually alongside their respective variables before')
st.markdown('developing kpis to better understand the data and standarize the way we measure and predict it.')

st.markdown('## EDA')
st.markdown('For the EDA we first began by looking at the timestamp variable, which will be a comonality among our three sections.')
st.markdown('We decided that we were goign to first change the datatype to datetime for better timeseries manipulation. Another primary step we took was to resample our data to')
st.markdown('have it once every minute instead of every 10 seconds. We then looked into the distribution of our data and took appropiate action depending on the distribution of our data. It was decided that we would try a general solution, but focus also on individual servers (nodes)')
st.markdown('Each section had a different approach as we prefered to focus on individual anomalies between systems. We believe that with enough time a good anomaly detection model could be developed for the combination of multiple sections.')

st.markdown('## Anomaly Detection')
st.markdown('We decided to utilize Isolated Forests for our anomaly detection on both memory and cpu as we believe it worked well  with the distribution of our data. For network we decided to take a different approach.')
st.markdown('Network had different distributions and also only two variables, incoming and outgoing traffic. We decided that for this we could try and implement a paper from Microsoft which stated that you could do anomaly detection with CNNs by first changing the signal to a saliency map with spectral residuals.')
st.markdown('This technique by Microsoft as utilizes what microsoft calls a SR-CNN, which is what we used. For the sake of time we utilize a library which had already built the initial foundation of the model')
st.markdown('We did this because we believe that we had enough justification to use it but not enough time to implement it and still finish with the other two sections.')