IF(WIN32)
    file (GLOB VISUALIZER_HEADER "*.h")
    file(GLOB VISUALIZER_SRC "*.cpp" "*.c")
ENDIF()
#message( ${CAPTURE_HEADER} )
#message( ${CMAKE_CURRENT_SOURCE_DIR} )


add_library(
    visualizer_lib 
    ${VISUALIZER_SRC} 
    ${VISUALIZER_HEADER} 
)


IF(WIN32 AND MSVC)
    target_link_libraries (
    visualizer_lib
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
    ${HELPER_LIBS}

)
ENDIF()


SET_TARGET_PROPERTIES(visualizer_lib PROPERTIES LINKER_LANGUAGE CXX)