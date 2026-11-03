idf_component_register(SRCS "main.c" "policy.c" "obs_builder.c" "sensors.c" "host_link.c"
                       INCLUDE_DIRS "."
                       REQUIRES freertos lwip)
