################################################################################
# UNIVERSIDADE FEDERAL DE CATALÃO (UFCAT)
# WANDERLEI MALAQUIAS PEREIRA JUNIOR,                  ENG. CIVIL / PROF (UFCAT)
# MARCOS NAPOLEÃO RABELO,                              ENG. CIVIL / PROF (UFCAT)
# DAVIDSON DE OLIVEIRA FRANÇA JUNIOR,                    ENG. CIVIL / PROF (UNA)
# GABRIEL BERNARDES CARVALHO,                                 ENG. CIVIL (UFCAT)
# JOSÉ VITOR CARVALHO SILVA,                                  ENG. CIVIL (UFCAT)
# MURILO CARNEIRO RODRIGUES,                                  ENG. CIVIL (UFCAT)
################################################################################


################################################################################
# DESCRIÇÃO ALGORITMO:
# BIBLIO. DE USO COMUM PARA ALGORITMOS DE ELEMENTOS FINITOS DESENVOL-
# VIDA PELO GRUPO DE PESQUISA E ESTUDOS EM ENGENHARIA (GPEE)
################################################################################


################################################################################
# BIBLIOTECAS NATIVAS PYTHON
import numpy as np

################################################################################
# BIBLIOTECAS DESENVOLVEDORES GPEE
import FINITO_TOOLBOX.FINITO_COMMON_LIBRARY as FINITO_CL

# PROGRAMA DE ELEMENTOS FINITO PARA BARRAS COM DOIS NÓS (1 NÓ POR EXTREMIDADE)
def MEF1D(**kwargs):
    """ 
    This function performs structural analysis of bars with 2 nodes (1 at each end)
    
    Input:
    All inputs kwargs arguments type
    FILENAME: Structural dataset (.txt extension)
    DICTIONARY: Structural dataset (Python dictionary)

    Output:
    RESULTS: Structural analysis results by element (Python dictionary)
    """
    # READ INPUT MEF 1D
    FILENAME = kwargs.get('FILENAME')
    DICTIONARY = kwargs.get('DICTIONARY')
    if FILENAME:
        TYPE_SOLUTION, TYPE_ELEMENT, N_NODES, N_MATERIALS, N_SECTIONS, N_ELEMENTS, N_DOFPRESCRIPTIONS, N_ELEMENTSLOADED, N_NODESLOADED, N_SPRINGS, COORDINATES, ELEMENTS, MATERIALS, SECTIONS, PRESCRIPTIONS, ELEMENT_EXTERNAL_LOAD, NODAL_EXTERNAL_LOAD, SPRINGS = FINITO_CL.GET_VALUE_FROM_TXT_MEF1D_FINITO(FILENAME)
    else:
        TYPE_SOLUTION, TYPE_ELEMENT, N_NODES, N_MATERIALS, N_SECTIONS, N_ELEMENTS, N_DOFPRESCRIPTIONS, N_ELEMENTSLOADED, N_NODESLOADED, N_SPRINGS, COORDINATES, ELEMENTS, MATERIALS, SECTIONS, PRESCRIPTIONS, ELEMENT_EXTERNAL_LOAD, NODAL_EXTERNAL_LOAD, SPRINGS = FINITO_CL.GET_VALUE_FROM_DICT_MEF1D_FINITO(DICTIONARY)
    # INDEX ASSEMBLY
    N_DOFSNODE, N_NODESELEMENT, DOFS_ACTIVE, DOFS_LOCAL, AUX_1, AUX_2, N_DOFSELEMENT = FINITO_CL.INDEX_ASSEMBLY(TYPE_ELEMENT)
    # GLOBAL DOF ASSEMBLY
    DOF_GLOBALNODAL = FINITO_CL.DOF_GLOBAL_ASSEMBLY(TYPE_ELEMENT, N_DOFSNODE, N_NODES)
    # DOFS TOTALS, PRESCRIPTIONS AND FREES
    DOF_GLOBAL, N_DOFSGLOBAL = FINITO_CL.TOTAL_DEGREE_FREEDOM(N_DOFSNODE, N_NODES)
    DOF_PRESCRIPTIONS, DOF_PRESCRIPTIONSVALUE, N_DOFSPRESCRIPTIONS = FINITO_CL.PRESCRIPTIONS_DEGREE_FREEDOM(PRESCRIPTIONS, DOF_GLOBALNODAL, N_DOFSNODE)
    DOF_FREE, N_DOFSFREE = FINITO_CL.FREE_DEGREE_FREEDOM(DOF_PRESCRIPTIONS, DOF_GLOBAL)
    # NODAL LOAD CONTRIBUTION
    DOF_NODALFORCE = FINITO_CL.CONTRIBUTION_NODAL_EXTERNAL_LOAD(NODAL_EXTERNAL_LOAD, N_DOFSGLOBAL, DOF_GLOBALNODAL)
    F_EXT = DOF_NODALFORCE
    # GLOBAL STIFFNES AND INTERNAL LOADS ASSEMBLY
    K_G = np.zeros((N_DOFSGLOBAL, N_DOFSGLOBAL))
    F_INT = np.zeros((N_DOFSGLOBAL, 1))
    # HINGED ELEMENTS
    HINGES = FINITO_CL.HINGED_PROPERTIES(ELEMENTS)
    for I_ELEMENT in range(N_ELEMENTS):
        # ELEMENTS PROPERTIES
        MATERIAL_IELEMENT = FINITO_CL.MATERIALS_PROPERTIES_0(ELEMENTS, MATERIALS, I_ELEMENT, AUX_1)
        SECTION_IELEMENT = FINITO_CL.GEOMETRIC_PROPERTIES_0(COORDINATES, ELEMENTS, SECTIONS, I_ELEMENT, AUX_2)
        # ELEMENT STIFFNESS LOCAL AXIS
        K_IELEMENT = FINITO_CL.ELEMENT_STIFFNESS_0(TYPE_ELEMENT, SECTION_IELEMENT, MATERIAL_IELEMENT, HINGES[I_ELEMENT, :])
        # ROTATION
        R_IELEMENT = FINITO_CL.ELEMENT_ROTATION(TYPE_ELEMENT, SECTION_IELEMENT)
        # ELEMENT STIFFNESS GLOBAL AXIS
        K_IELEMENTGLOBAL = np.dot(np.dot(np.transpose(R_IELEMENT), K_IELEMENT), R_IELEMENT)
        # GOBAL DOF I_ELEMENT
        DOF_GLOBALIELEMENT = FINITO_CL.GLOBAL_DOF_ELEMENT(N_NODESELEMENT, N_DOFSNODE, DOF_GLOBALNODAL, ELEMENTS, I_ELEMENT)
        # GLOBAL STIFFNESS
        K_GCONTRIBUITION = FINITO_CL.GLOBAL_STIFFNESS(N_DOFSGLOBAL, DOF_GLOBALIELEMENT, K_IELEMENTGLOBAL)
        K_G = K_G + K_GCONTRIBUITION
    # SPRING CONTRIBUTION
    if N_SPRINGS > 0:
        SPRING_INDEX, SPRING_VALUES = FINITO_CL.SPRING_CONTRIBUTION(N_DOFSNODE, SPRINGS, N_SPRINGS)
        for K_COUNT in range(N_SPRINGS):
            SPRING_VALUE = SPRING_VALUES[K_COUNT]
            INDEX_DOF = SPRING_INDEX[K_COUNT]
            K_G[INDEX_DOF, INDEX_DOF] = K_G[INDEX_DOF, INDEX_DOF] + SPRING_VALUE
    # DISPLACEMENT SOLUTION: CONDENSE PROCEDURE
    if TYPE_SOLUTION == 0:
        # CONDENSE DISPLACEMENTS, FORCES AND STIFFNESS MATRIX
        U_PP = FINITO_CL.CONDENSE_PRESCRIBED_GLOBAL_DISPLACEMENT(DOF_PRESCRIPTIONSVALUE, N_DOFSPRESCRIPTIONS)
        F_FF = FINITO_CL.CONDENSE_FREE_GLOBAL_FORCES(F_EXT, DOF_FREE, N_DOFSFREE)
        K_FF = FINITO_CL.CONDENSE_GLOBAL_FREE_STIFFNESS(K_G, DOF_FREE, N_DOFSFREE)
        K_PF = FINITO_CL.CONDENSE_PRESCRIBED_FREE_GLOBAL_STIFFNESS(K_G, DOF_FREE, N_DOFSFREE, DOF_PRESCRIPTIONS, N_DOFSPRESCRIPTIONS)
        # SOLUTION
        K_FFINVERSE = np.linalg.pinv(K_FF, rcond=1e-15)
        U_FF = np.dot(K_FFINVERSE, F_FF - np.dot(np.transpose(K_PF), U_PP))
        U_G = FINITO_CL.ASSEMBLY_TOTAL_DISPLACEMENT(U_FF, U_PP, N_DOFSGLOBAL, DOF_PRESCRIPTIONS, DOF_FREE)
    # DISPLACEMENT SOLUTION: ZERO AND ONE PROCEDURE
    elif TYPE_SOLUTION == 1:
        pass
    # INTERNAL LOADS
    # FRAME DIVISION
    DIV = 11 
    # START EMPTY DICTIONARY RESULTS
    RESULTS = [{'X': np.empty(DIV), 'UX': np.empty(DIV), 'UY': np.empty(DIV), 'UZ': np.empty(DIV), 'N': np.empty(DIV), 'V': np.empty(DIV), 'M': np.empty(DIV), 'ID_ELEMENT': J_COUNT} for J_COUNT in range(N_ELEMENTS)]
    for J_ELEMENT in range(N_ELEMENTS):
        # ELEMENT PROPERTIES: MATERIAL AND SECTION
        MATERIAL_JELEMENT = FINITO_CL.MATERIALS_PROPERTIES_0(ELEMENTS, MATERIALS, J_ELEMENT, AUX_1)
        SECTION_JELEMENT = FINITO_CL.GEOMETRIC_PROPERTIES_0(COORDINATES, ELEMENTS, SECTIONS, J_ELEMENT, AUX_2)
        # ELEMENT STIFFNESS LOCAL AXIS
        K_JELEMENT = FINITO_CL.ELEMENT_STIFFNESS_0(TYPE_ELEMENT, SECTION_JELEMENT, MATERIAL_JELEMENT, HINGES[J_ELEMENT, :])
        # ROTATION
        R_JELEMENT = FINITO_CL.ELEMENT_ROTATION(TYPE_ELEMENT, SECTION_JELEMENT)
        # GOBAL DOF J_ELEMENT
        DOF_GLOBALJELEMENT = FINITO_CL.GLOBAL_DOF_ELEMENT(N_NODESELEMENT, N_DOFSNODE, DOF_GLOBALNODAL, ELEMENTS, J_ELEMENT)
        # ELEMENT GLOBAL DISPLACEMENTS
        U_GJELEMENT = FINITO_CL.CONDENSE_GLOBAL_ELEMENT_DISPLACEMENTS(U_G, N_DOFSELEMENT, DOF_GLOBALJELEMENT)
        # ELEMENT LOCAL DISPLACEMENTS
        U_JELEMENT = np.dot(R_JELEMENT, U_GJELEMENT)
        # INTERNAL FORCE IN J_ELEMENT (NODE 1 AND NODE 2 [X, Y, Z])
        F_ELINT = np.dot(K_JELEMENT, U_JELEMENT)
        # INTERNAL FORCES IN ELEMENT (DIVISION)
        for I_COUNT in range(DIV):
            # DISTANCE IN LOCAL AXIS
            X = SECTION_JELEMENT[0] * (I_COUNT) / (DIV - 1)
            if I_COUNT == 0:
                U_X = U_GJELEMENT[0, 0]
                U_Y = U_GJELEMENT[1, 0]
                U_Z = U_GJELEMENT[2, 0]
            elif I_COUNT == DIV - 1:
                U_X = U_GJELEMENT[3, 0]
                U_Y = U_GJELEMENT[4, 0]
                U_Z = U_GJELEMENT[5, 0]
            elif (I_COUNT != 0 and I_COUNT != DIV - 1):
                U_X = -1989
                U_Y = -1989
                U_Z = -1989
            # INTERNAL LOADS: AXIAL, SHEAR AND BENDING MOMENT 
            N = -F_ELINT[0]
            V = F_ELINT[1]
            M = -F_ELINT[2] + F_ELINT[1] * X         
            # SAVE RESULTS IN DICTIONARY
            RESULTS[J_ELEMENT]['X'][I_COUNT] = X
            RESULTS[J_ELEMENT]['UX'][I_COUNT] = U_X
            RESULTS[J_ELEMENT]['UY'][I_COUNT] = U_Y
            RESULTS[J_ELEMENT]['UZ'][I_COUNT] = U_Z
            RESULTS[J_ELEMENT]['N'][I_COUNT] = N
            RESULTS[J_ELEMENT]['V'][I_COUNT] = V
            RESULTS[J_ELEMENT]['M'][I_COUNT] = M
    return RESULTS


def MEF2D(**kwargs):
    """ 
    This function performs structural analysis via finite elements 
    considering flat triangular elements (CST, LST*, QST*)
    
    * Under development

    Input:
    All inputs kwargs arguments type
    FILENAME: Structural dataset (.txt extension)
    DICTIONARY: Structural dataset (Python dictionary)

    Output:
    RESULTS: Structural analysis results by element (Python dictionary)
    """
    # Read input file
    FILENAME = kwargs.get('FILENAME')
    DICTIONARY = kwargs.get('DICTIONARY')
