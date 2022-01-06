
if("${BUILD_TEST}")

    target_sources(shamrock_test
    PRIVATE
    
    PUBLIC
        test_scheduler.hpp 
    )


endif()
