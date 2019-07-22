
set(proj SphericalRidgelets)

# Set dependency list
set(${proj}_DEPENDENCIES ITK Eigen)

# Include dependent projects if any
ExternalProject_Include_Dependencies(${proj} PROJECT_VAR proj DEPENDS_VAR ${proj}_DEPENDENCIES)

if(${CMAKE_PROJECT_NAME}_USE_SYSTEM_${proj})
  unset(SphericalRidgelets_DIR CACHE)
  find_package(SphericalRidgelets REQUIRED)
endif()

# Sanity checks
if(DEFINED SphericalRidgelets_DIR AND NOT EXISTS ${SphericalRidgelets_DIR})
  message(FATAL_ERROR "SphericalRidgelets_DIR variable is defined but corresponds to nonexistent directory")
endif()

if(NOT DEFINED SphericalRidgelets_DIR AND NOT ${CMAKE_PROJECT_NAME}_USE_SYSTEM_${proj})

  ExternalProject_SetIfNotDefined(
    ${CMAKE_PROJECT_NAME}_${proj}_GIT_TAG
    "55470c2110fd5f5a77a024513d45486d41f20646"
    QUIET
  )
  
  ExternalProject_SetIfNotDefined(
    ${CMAKE_PROJECT_NAME}_${proj}_GIT_REPOSITORY 
    "${git_protocol}://github.com/rmukh/spherical_ridgelets.git"
    QUIET
  )

  set(EP_SOURCE_DIR ${CMAKE_BINARY_DIR}/${proj})
  set(EP_BINARY_DIR ${CMAKE_BINARY_DIR}/${proj}-build)
  set(EP_INSTALL_DIR ${CMAKE_BINARY_DIR}/${proj}-install)

  ExternalProject_Add(${proj}
    ${${proj}_EP_ARGS}
    GIT_REPOSITORY "${${CMAKE_PROJECT_NAME}_${proj}_GIT_REPOSITORY}"
    GIT_TAG "${${CMAKE_PROJECT_NAME}_${proj}_GIT_TAG}"
    SOURCE_DIR ${EP_SOURCE_DIR}
    BINARY_DIR ${EP_BINARY_DIR}
    INSTALL_DIR ${EP_INSTALL_DIR}
    CMAKE_CACHE_ARGS
      -DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}
      -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
      -DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}
      -DCMAKE_C_FLAGS:STRING=${ep_common_c_flags}
      -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
      -DITK_DIR:PATH=${ITK_DIR}
      -DEigen3_DIR:PATH=${Eigen_DIR}
      -DSPH_DIR:PATH=${EP_SOURCE_DIR}/Ridgelets/SphericalRidgelets
      -DCMAKE_INSTALL_LIBDIR:PATH=${EP_INSTALL_DIR}/lib
    DEPENDS
      ${${proj}_DEPENDENCIES}
    )

  set(SphericalRidgelets_DIR ${EP_INSTALL_DIR}/lib/cmake/SphericalRidgelets)
  set(SphericalRidgelets_ROOT ${EP_INSTALL_DIR})
  set(SphericalRidgelets_INCLUDE_DIR ${EP_INSTALL_DIR}/include)
  if(WIN32)
    set(SphericalRidgelets_LIBRARY ${EP_INSTALL_DIR}/lib/Spherical_Ridgelets.lib)
  else()
    set(SphericalRidgelets_LIBRARY ${EP_INSTALL_DIR}/lib/libSpherical_Ridgelets.a)
  endif()

  #-----------------------------------------------------------------------------
  # Launcher setting specific to build tree

  # library paths
  set(${proj}_LIBRARY_PATHS_LAUNCHER_BUILD ${SphericalRidgelets_DIR}/bin/<CMAKE_CFG_INTDIR>)
  mark_as_superbuild(
    VARS ${proj}_LIBRARY_PATHS_LAUNCHER_BUILD
    LABELS "LIBRARY_PATHS_LAUNCHER_BUILD" "PATHS_LAUNCHER_BUILD"
    )

else()
  ExternalProject_Add_Empty(${proj} DEPENDS ${${proj}_DEPENDENCIES})
endif()

mark_as_superbuild(
  VARS
    SphericalRidgelets_ROOT:PATH
    SphericalRidgelets_DIR:PATH
    SphericalRidgelets_INCLUDE_DIR:PATH
    SphericalRidgelets_LIBRARY:FILEPATH
  LABELS "FIND_PACKAGE"
  )
