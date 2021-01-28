# Predicting County-Level Mask Wearing

This model is an initial prototype that attempts to forecast mask wearing based on the interaction of static determinants of health behaviors, such as demographic, economic, and social indicators, with information exposure from news and social media. The long-term goal would be to develop a model that is responsive to a changing information environment. Specifically, can we measure how changing information in news and media influences peoples' decision to wear a mask or not at the county-level?

## Integrating realtime information data to predict behavior

The secondary goal of this model is to develop a methodology for integrating realtime information flows with contextual information to predict and/or forecast how different groups of people might change their attitudes, beliefs, and/or behaviors based on an evolving information ecosystem. This type of work could be useful to quickly detect any potential changes in human behavior and help, for example, public health practitioners to better allocate resources, design more targeted health communication campaigns, etc. 

## Feature Data

This MVP uses as its feature set numeric inputs from the CDC Social Vulnerability Index, Measure of America Youth Disconnection Index, and Apple mobility data at the county-level, combined with county-level geolocated tweets and state geotagged Covid-related news. 

The *CDC Social Vulnerability Index* data is mostly complete, so the only preprocessing I did was to apply a MinMax scaler to scale the data since each dimensions typically represents a different kind of measurement. I also converted the county FIPS code and state in categorical variables.
https://www.atsdr.cdc.gov/placeandhealth/svi/data_documentation_download.html

For the *Measure of America Youth Disconnection Index*, there are a number of counties, particularly those that are smaller and/or more rural, that did not have data. In that case, I replaced those missing values with the mean for each county's respective state.
http://www.measureofamerica.org/DYinteractive/#County

I filtered the *Apple mobility* data for the two weeks prior to NYT face mask survey dates. For counties that lacked data, I filled those missing data with the country-wide mean for that day.
https://covid19.apple.com/mobility

Tweets from June 15 - July 15 were extracted from the *Coronvavirus Tweets Dataset*, created by Rabindra Lamsal, geolocated to the relevant US county, tokenized, and aggregated into a set of tokens for each county.
https://ieee-dataport.org/open-access/coronavirus-covid-19-geo-tagged-tweets-dataset

Finally, I used a dataset of U.S. political news articles I had compiled for another project, did some light "covid" keyword filtering for relevant text, applied some light keyword matching for state names, tokenized, and then aggregated at the state level.
https://github.com/AschHarwood/predicting_attitudes_from_news

Text data collection and processing is quite sloppy for this particular version, but I wanted to spend some time simply learning how to build multi-input Keras models.

### Shape of the Data

The final feature data set is composed of 3142 rows, one row for each U.S. county. You can view an overview of the data via this Pandas Profile report.
http://htmlpreview.github.io/?https://github.com/AschHarwood/covid_county_mask_predictions/blob/main/notebooks/dataset_profile.html


## Target Data

The target for this model is binary classification about whether more or less than 50 percent of the population for each county wear's a mask. It's derived from the New York Times July 2020 survey into mask wearing.
https://github.com/nytimes/covid-19-data/tree/master/mask-use

## Model Structure

The model's architecture consists of a multi-input feed forward neural net built with Keras' functional API. To preprocess the text data, I used word embeddings created by a pretrained model custom build for covid-related content.

## Results

The results are quite promising. With some light cross-fold validation, the model returned an average 78 percent accuracy. Decent results shouldn't be too surprising, since, theoretically, there should be a relationship between socioeconomic determinants of health, information exposure, and mask wearing. Essentially, the idea is to quantify the social ecological model, which is a framework for understanding what influences health-related behaviors.

It is worth noting that the news and/or tweet models alone do not do very well...yet. On the other hand, the numeric-only model returns somewhere between 72 - 75 accuracy on a test set. So the text data, even as spotty as it is, still seems to be adding value to the model.

## Next Steps

- Secure additional relevant text data (news, television transcripts, social media posts, statements from relevant political leaders)
- Identity an additional target mask wearing data at another point in time
- Add additional indicators that likely have a bearing on mask wearing, such political or religious affiliation, vaccine coverage rates, etc



