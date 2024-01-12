#Performing analysis on Variable 1 : CPI with Gamma (ie has seasonality)

#import libraries 
library('forecast')
library('stats')

#read/attach data
data=read.csv("/Users/katelassiter/Downloads/303/312/497T/AllVariablesforR.csv")
attach(data)

#generate time series
CPI_Monthly=ts(CPI, frequency=12)
print(CPI_Monthly)
plot(CPI_Monthly)

#Create size restrictions
N=length(CPI_Monthly)
WithinSampleLength=N-27
WithinSample=ts(CPI_Monthly[1:WithinSampleLength],frequency=12)
print(WithinSample)


# Now we will be performing a holt's method test 
HoltWintersMethod=HoltWinters(WithinSample,alpha=NULL,beta=NULL,gamma=NULL)

#This gives you the smoothing parameters
print(HoltWintersMethod)

#Finding the sum of squared errors and RMSE

SSE = HoltWintersMethod$ SSE
LengthWithLostObservations=WithinSampleLength-12
RMSE=sqrt(SSE/LengthWithLostObservations)
print(RMSE)


#Gives basically nothing
summary(HoltWintersMethod)

#Fitting the within sample values with Holt's Method
WithinXhats=fitted(HoltWintersMethod)
print(WithinXhats)


#Splitting actual into test(actual) vs train(xhat) sets
CPIPostSample=ts(CPI_Monthly)
WithinTest=CPIPostSample[13:WithinSampleLength]
print(WithinTest)
PostTest=CPIPostSample[(WithinSampleLength+1):N]
print(PostTest)


#Getting forecasts
PostXhats=predict(HoltWintersMethod,n.ahead=27)
print(PostXhats)

#Finding the RMSE
RMSE=function(PostTest,PostXhats){
	sqrt(mean((PostTest-PostXhats)^2))
}
print(RMSE(PostTest,PostXhats))


_______________________________________________________________________________

#Performing analysis on Variable 2 : Gov Expenditure without Gamma

#import libraries 
library('forecast')
library('stats')

#read/attach data
data=read.csv("/Users/katelassiter/Downloads/303/312/497T/Variable2.csv")
attach(data)

#generate time series
Expenditure_Quarterly=ts(Expenditure, frequency=4)
print(Expenditure_Quarterly)
plot(Expenditure_Quarterly)

#Create size restrictions
N=length(Expenditure_Quarterly)
WithinSampleLength=N-8
WithinSample=ts(Expenditure_Quarterly[1:WithinSampleLength],frequency=4)
print(WithinSample)


# Now we will be performing a holt's method test 
HoltMethod=HoltWinters(WithinSample,alpha=NULL,beta=NULL,gamma=FALSE)

#This gives you the smoothing parameters
print(HoltMethod)

#Finding the sum of squared errors and RMSE
SSE = HoltMethod$ SSE
LengthWithLostObservations=WithinSampleLength-2
RMSE=sqrt(SSE/LengthWithLostObservations)
print(RMSE)


#Gives basically nothing
summary(HoltMethod)

#Fitting the within sample values with Holt's Method
WithinXhats=fitted(HoltMethod)
print(length(WithinXhats))
print(WithinXhats)


#Splitting actual into test(actual) vs train(xhat) sets
ExpenditurePostSample=ts(Expenditure_Quarterly)
WithinTest=ExpenditurePostSample[3:WithinSampleLength]
print(WithinTest)
PostTest=ExpenditurePostSample[(WithinSampleLength+1):N]
print(PostTest)


#Getting forecasts
PostXhats=predict(HoltMethod,n.ahead=8)
print(PostXhats)

#Finding the RMSE
RMSE=function(PostTest,PostXhats){
	sqrt(mean((PostTest-PostXhats)^2))
}
print(RMSE(PostTest,PostXhats))
