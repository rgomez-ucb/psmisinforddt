setwd("/Users/ruben/Desktop/final-project-repo-team-trust/data") #set your working directory
p_load(ggplot2,dplyr) #install packages

reddit_data <-read.csv("joined_data.csv")
head(reddit_data)

# create counts for each year
year_counts <- reddit_data %>%
  group_by(year) %>%
  summarise(count = n())

head(year_counts)

# plot line chart

ggplot(year_counts, aes(x = year, y = count)) +
  geom_line(color = "steelblue", size = 1.2) +
  geom_point(color = "darkred", size = 2) +
  labs(title = "Number of Posts Over Years",
       x = "Year",
       y = "Count") +
  theme_minimal()

# Checking for missed yearly data 2018-2023
table(reddit_data$year)

# Create timestamps using submission column created_utc_x and commment column created_utc_y
reddit_data$year <- ifelse(
  is.na(reddit_data$created_utc_y),
  format(as.POSIXct(reddit_data$created_utc_x, origin = "1970-01-01", tz = "UTC"), "%Y"),
  format(as.POSIXct(reddit_data$created_utc_y, origin = "1970-01-01", tz = "UTC"), "%Y")
)

table(reddit_data$year)

# Count rows per year (Posts + Comments)
year_counts <- reddit_data %>%
  group_by(year) %>%
  summarise(count = n()) %>%
  mutate(
    year = as.numeric(year),
    count = as.numeric(count),
    count_rounded = round(count / 10) * 10
  )

# plot counts
ggplot(year_counts, aes(x = as.numeric(year), y = count)) +
  geom_line(color = "skyblue", size = 1.2) +
  geom_point(color = "red", size = 3) +
  geom_text(aes(label = count), vjust = -0.6, size = 3.5) +
  scale_x_continuous(
    breaks = seq(min(year_counts$year), max(year_counts$year), by = 1)  # show every year
  ) +
  scale_y_continuous(
    breaks = seq(0, max(year_counts$count), by = 2000)  # adjust to your range
  ) +
  labs(
    title = "Total Posts and Comments by Year in r/PoliticalDiscussion",
    x = "Year",
    y = "Count"
  ) +
  theme_minimal()

