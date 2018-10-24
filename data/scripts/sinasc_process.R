## script to process the SINASC data
##---------------------------------------------------------

Sys.setlocale("LC_ALL", "C")
# Sys.setenv(R_MAX_VSIZE = 64e9)
# Sys.getenv("R_MAX_VSIZE")

library(cidacsdict)
library(tidyverse)

snsc_dict <- subset(cdcs_dict, db == "SINASC")

newvar <- data_frame(
  id = 1,
  db_name = "SINASC",
  db_name_en = "SINASC",
  name = "codestab",
  name_en = "health_estbl_code",
  label = "Código de estabelecimento de saúde",
  label_en = "Health establishment code",
  label_google_en = "Health establishment code",
  map = NA,
  map_en = NA,
  type = "character",
  presence = "",
  presence_en = "",
  comments_en = "",
  db = "SINSAC",
  map_en_orig = NA,
  map_orig = NA
)

snsc_dict <- bind_rows(snsc_dict, newvar)

ff <- list.files("data/discovered/_raw/datasus/SINASC/DNRES", full.names = TRUE)
# restrict to 2006+
# ff <- ff[!grepl("2001|2002|2003|2004|2005", ff)]

res <- vector(length = length(ff), mode = "list")

a <- read_datasus(ff[405]) %>%
  transform_data(dict = snsc_dict, quiet = TRUE)

for (ii in seq_along(ff)) {
  message(ii)
  res[[ii]] <- read_datasus(ff[ii]) %>%
    transform_data(dict = snsc_dict, quiet = TRUE) %>%
    clean_data()
}

table(sapply(res, ncol))

snsc <- bind_rows(res)

ncol(snsc)
nrow(snsc_dict)

# look at variables in the dictionary that aren't in the public SINASC data
snsc_dict %>%
  filter(! name_en %in% names(snsc)) %>%
  select(name, name_en, label_en)
#   name       name_en           label_en
# --------------------------------------------------------------
# cepnasc    birth_postal_code Zip code of the birth place
# codestocor birth_fu_code     Federation Unit Code of the birth
# cepres     res_postal_code   Zip code of the residence

# snsc <- snsc %>% tibble::as.tibble()
# save(snsc, file = "data/snsc_all2.Rdata")
# load("data/discovered/snsc_all.Rdata")

kitools::data_publish(snsc, name = "snsc_2001-2015",
  desc = "Public individual-level SINASC data from 2001-2015, with variables that intersect with the CIDACS GCE data dictionary.")

snsc2 <- filter(snsc, birth_year >= 2011)

kitools::data_publish(snsc2, name = "snsc_2011-2015",
  desc = "Public individual-level SINASC data from 2011-2015, with variables that intersect with the CIDACS GCE data dictionary.")
