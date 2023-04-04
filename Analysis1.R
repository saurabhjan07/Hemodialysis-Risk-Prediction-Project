library (plyr)
library(ggplot2)

dataset1 <- read.csv("./HR/Event1.csv")          # Only Events_Patients Data
dataset2 <- read.csv("./HR/NoEvent1.csv")     # No Events_Patients Data                
dataset3 <- read.csv("./HR/Death1.csv")                 # Five Events_Patients Data - DEATH 
dataset4 <- read.csv("./HR/EV1.csv")             # Five Events_Patients Data - Emergency
dataset5 <- read.csv("./HR/EVIP1.csv") # Five Events_Patients Data - EmergencyAndInpatient
dataset6 <- read.csv("./HR/IP1.csv")             # Five Events_Patients Data - Inpatient 
dataset7 <- read.csv("./HR/MS1.csv")           # Five Events_Patients Data - MuscleSpasm 




# Dataframe combined No Event & Death Event: Date+ID+HR

dataset2$Event <- "NoEvent"
dataset3$Event <- "Death"

common_cols <- intersect(colnames(dataset2), colnames(dataset3))

df1 <- rbind(
  subset(dataset2, select = common_cols), 
  subset(dataset3, select = common_cols)
)



gg <- ggplot(dataset2, aes(x=Date, y=HR)) 
gg1 = gg + geom_point(size=1, shape=3, color='green', stroke=2)
print(gg1)

gg2 <- ggplot(dataset3, aes(x=Date, y=HR)) 
gg3 = gg2 + geom_point(size=1, shape=3, color='red', stroke=2)
print(gg3)

gg <- ggplot() + 
  geom_point(data=dataset2, aes(x=Date, y=HR), color='green', stroke=1, size=1) + 
  geom_point(data=dataset3, aes(x=Date, y=HR), color='red', stroke=1,size=1)
print(gg)

gg <- ggplot() + 
  geom_point(data=dataset2, aes(x=Id, y=HR), color='green', stroke=1, size=1) + 
  geom_point(data=dataset3, aes(x=Id, y=HR), color='red', stroke=1,size=1)
print(gg)

#######################HF Plot ############################

dataset2 <- read.csv("./HRV/NoEvent1.csv")               # No Events_Patients Data                
dataset3 <- read.csv("./HRV/Death1.csv")                 # Five Events_Patients Data - DEATH 

gg <- ggplot() + 
  geom_point(data=dataset2, aes(x=Date, y=HF), color='green', stroke=1, size=1) + 
  geom_point(data=dataset3, aes(x=Date, y=HF), color='red', stroke=2,size=1)
print(gg)

gg <- ggplot(dataset2, aes(x=Date, y=HF)) 
gg1 = gg + geom_point(size=1, shape=1, color='blue', stroke=1)
print(gg1)

gg <- ggplot(dataset3, aes(x=Date, y=HF)) 
gg1 = gg + geom_point(size=1, shape=1, color='red', stroke=1)
print(gg1)

#HR_NEvsDeath_FFM
gg <- ggplot(df1, aes(x=Date, y=HR)) 
gg1 = gg + geom_point(size=1, shape=3, color=ifelse(df1$Event=='Death','red','green'), stroke=2)
print(gg1)

#HR_NEvsDeath_LFM

gg <- ggplot(df1, aes(x=Date, y=HR_LFM)) 
gg1 = gg + geom_point(size=1, shape=3, color=ifelse(df1$Event=='Death','red','green'), stroke=2)
print(gg1)


#RR_NEvsDeath_FFM

gg <- ggplot(df1, aes(x=Date, y=RR)) 
gg1 = gg + geom_point(size=1, shape=3, color=ifelse(df1$Event=='Death','red','green'), stroke=2)
print(gg1)

#RR_NEvsDeath_LFM

gg <- ggplot(df1, aes(x=Date, y=RR_LFM)) 
gg1 = gg + geom_point(size=1, shape=ifelse(df1$Event=='Death',1,2), color=ifelse(df1$Event=='Death','red','green'), stroke=2)
print(gg1)

#MovementDensity_NEvsDeath_FFM

gg <- ggplot(df1, aes(x=Date, y=MovementDensity)) 
gg1 = gg + geom_point(size=1, shape=ifelse(df1$Event=='Death',1,2), color=ifelse(df1$Event=='Death','red','green'), stroke=2)
print(gg1)

#MovementDensity_NEvsDeath_LFM

gg <- ggplot(df1, aes(x=Date, y=MovementDensity_LFM)) 
gg1 = gg + geom_point(size=1, shape=ifelse(df1$Event=='Death',1,2), color=ifelse(df1$Event=='Death','red','green'), stroke=2)
print(gg1)


#HF_NEvsDeath_FFM

gg <- ggplot(df1, aes(x=Date, y=HF)) 
gg1 = gg + geom_point(size=1, shape=ifelse(df1$Event=='Death',1,2), color=ifelse(df1$Event=='Death','red','green'), stroke=2)
print(gg1)

#HF_NEvsDeath_LFM

gg <- ggplot(df1, aes(x=Date, y=HF_LFM)) 
gg1 = gg + geom_point(size=1, shape=ifelse(df1$Event=='Death',1,2), color=ifelse(df1$Event=='Death','red','green'), stroke=2)
print(gg1)

#LF_NEvsDeath_FFM

gg <- ggplot(df1, aes(x=Date, y=LF)) 
gg1 = gg + geom_point(size=1, shape=ifelse(df1$Event=='Death',1,2), color=ifelse(df1$Event=='Death','red','green'), stroke=2)
print(gg1)

#LF_NEvsDeath_LFM

gg <- ggplot(df1, aes(x=Date, y=LF_LFM)) 
gg1 = gg + geom_point(size=1, shape=ifelse(df1$Event=='Death',1,2), color=ifelse(df1$Event=='Death','red','green'), stroke=2)
print(gg1)

#LF/HF_NEvsDeath_FFM

gg <- ggplot(df1, aes(x=Date, y=LF/HF)) 
gg1 = gg + geom_point(size=1, shape=ifelse(df1$Event=='Death',1,2), color=ifelse(df1$Event=='Death','red','green'), stroke=2)
print(gg1)

#LF/HF_NEvsDeath_LFM

gg <- ggplot(df1, aes(x=Date, y=LF/HF_LFM)) 
gg1 = gg + geom_point(size=1, shape=ifelse(df1$Event=='Death',1,2), color=ifelse(df1$Event=='Death','red','green'), stroke=2)
print(gg1)

#VLF_NEvsDeath_FFM

gg <- ggplot(df1, aes(x=Date, y=VLF)) 
gg1 = gg + geom_point(size=1, shape=ifelse(df1$Event=='Death',1,2), color=ifelse(df1$Event=='Death','red','green'), stroke=2)
print(gg1)


#VLF_NEvsDeath_LFM

gg <- ggplot(df1, aes(x=Date, y=VLF_LFM)) 
gg1 = gg + geom_point(size=1, shape=ifelse(df1$Event=='Death',1,2), color=ifelse(df1$Event=='Death','red','green'), stroke=2)
print(gg1)


#(VLF+LF)/HF_NEvsDeath_FFM

gg <- ggplot(df1, aes(x=Date, y=(VLF+LF)/HF)) 
gg1 = gg + geom_point(size=1, shape=ifelse(df1$Event=='Death',1,2), color=ifelse(df1$Event=='Death','red','green'), stroke=2)
print(gg1)


#(VLF+LF)/HF_NEvsDeath_LFM

gg <- ggplot(df1, aes(x=Date, y=(VLF+LF)/HF_LFM)) 
gg1 = gg + geom_point(size=1, shape=ifelse(df1$Event=='Death',1,2), color=ifelse(df1$Event=='Death','red','green'), stroke=2)
print(gg1)

