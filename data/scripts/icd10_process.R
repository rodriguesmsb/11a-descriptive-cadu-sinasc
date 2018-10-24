icd10 <- read.csv("https://raw.githubusercontent.com/kamillamagna/ICD-10-CSV/master/codes.csv", stringsAsFactors = FALSE)
names(icd10) <- c("cat_code", "diag_code", "full_code", "desc_abbr", "desc_full", "cat_title")
icd10 <- as.tibble(icd10)

length(unique(icd10$cat_code))
# 17714
length(unique(icd10$full_code))
# 71670

# TODO: should fix these:
table(nchar(icd10$cat_code))
icd10$cat_code[nchar(icd10$cat_code) > 6]

table(nchar(icd10$diag_code))
table(nchar(icd10$full_code))

kitools::data_publish(icd10, name = "icd10_2018",
  desc = "Data frame of 2018 ICD-10 codes",
  used = "https://github.com/kamillamagna/ICD-10-CSV")

