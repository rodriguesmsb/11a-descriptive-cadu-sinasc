## script to process the SIM data
##---------------------------------------------------------

Sys.setlocale("LC_ALL", "C")
# Sys.setenv(R_MAX_VSIZE = 64e9)
# Sys.getenv("R_MAX_VSIZE")

library(cidacsdict)
library(tidyverse)

sim_dict <- subset(cdcs_dict, db == "SIM")

ff <- list.files("data/discovered/_raw/datasus/SIM/DORES", full.names = TRUE)

res <- vector(length = length(ff), mode = "list")

a <- read_datasus(ff[404]) %>%
  transform_data(dict = sim_dict, quiet = TRUE) %>%
  clean_data()

for (ii in seq_along(ff)) {
  message(ii)
  res[[ii]] <- read_datasus(ff[ii]) %>%
    transform_data(dict = sim_dict, quiet = TRUE) %>%
    clean_data()
}

table(sapply(res, ncol))

sim <- bind_rows(res)

# look at variables in the dictionary that aren't in the public SIM data
sim_dict %>%
  filter(! name_en %in% names(sim)) %>%
  select(name, name_en, label_en)
# name         name_en           label_en
#------------------------------------------------------------------
# unidade      age_units         Age unit
# cepres       postal_code       ZIP code
# codregres    resi_region_code  Area of residence code
# codpaisres   resi_country_code Country of residence code
# cepocor      death_postal_code Zip code of the place of death
# baiocor      death_nbhd        Neighborhood of the place of death
# codregocor   death_region_code Region code of occurrence
# semanagestac gest_weeks        Weeks of pregnancy

# table(sim$death_year, useNA = "always")
# sim <- filter(sim, death_year != 2005) # 2005 isn't complete

kitools::data_publish(sim, name = "sim_2001-2015",
  desc = "Public individual-level SIM data from 2001-2015, with variables that intersect with the CIDACS GCE data dictionary.")
