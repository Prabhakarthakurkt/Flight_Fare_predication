# Flight_Fare_predication
we will be predicating the fare of a flight a person has to give on inputting the data using the noraml machine learning techniques then will we see how can do the same thing with the halp of using Auto SK Learn which is a Auto Ml Library

## Context

we have often heared travelling saying Flight Ticket are often very unpredictable and are very hard to guess. if one might see a price today and price today and checks the same flight price today and checks the same flight price tommorow it's whole different story by then.

Let us create a Machine Learning Model which will help us in predicting the price of a flight on inputting some of the attributes. Here we will be provided with price of flight on inputting some of the atterbutes. Here we will be provided with prices of flight tickets for various airlines between the months of april and jun of 2019 and between various cities.

We will do the following things in our Notebooks

  # Data Analysis 
    feature Engineering 
    Feature Selection 
    Model Building using ML
    Model Building using Auto SK Learn(Auto ML)


So let's dive in and read our data, but first we will import all the necessary libraries

  # Reading our Data set

    df= pd.read_csv("flight data.csv")
    df.head()
    df.shape
    df1=pd.read_csv("flight test.csv")
    df1.head()
    df1.shape

we will combine both the data for the purpose of feature Engineering 

    total_df=df.append(df1,sort= False)
    total_df.tail()
    total_df.head()
    total_df.shape

# Understanding our Data

size of training set: 10683 records
size of test set: 2671 records


# Features

    Airline: The name of the airline.
    Data_of_journey: The data of journey
    source: The source from which the servies begins
    Destination: The destinations where the serive ends.
    Route: The route taken by the flight to reach the distination.
    Dep_Time: The time when the journey starts from the source
    Arrival_Time: Time of arrival at the destination .
    Duration: Total duration of the flight .
    Total_Stops: Total stops between the source and destination.
    Additional_info: Additional information about the flight 
    Price: The price of the ticket

  
# Data Analysis

we will be doing this on our train data only

    df['Airline'].value_counts()

Let's see the relation b/w airline and price fare 

    sns.catplot(y='price',x='airline',data=df.sort_values('price',ascending=False),
    kind="boxen",height=6,aspect=3)

jet airways business have the height fare

    df['source'].value_counts() 

Finding relation b/w source and price

    sns.catplot(y='price',x='source',data=df.df.sort_values('price',ascending=False),kind="boxen",height=6,aspect=3)
    df['Destination'].value_counts() 

Doing the same for destination

    
    
