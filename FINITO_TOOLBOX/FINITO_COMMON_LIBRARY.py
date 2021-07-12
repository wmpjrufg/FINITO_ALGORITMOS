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
# BIBLIO. FENON PARA FUNÇÕES COMUNS EM ELEMENTOS FINITOS DESENVOLVIDA PELO GRUPO
# DE PESQUISA E ESTUDOS EM ENGENHARIA (GPEE)
################################################################################

################################################################################
# BIBLIOTECAS NATIVAS PYTHON
import numpy as np

################################################################################
# BIBLIOTECAS DESENVOLVEDORES GPEE
def GET_VALUE_FROM_DICT_MEF1D_FINITO(DICTIONARY):
    TYPE_ELEMENT = DICTIONARY["TYPE_ELEMENT"]
    TYPE_SOLUTION = DICTIONARY["TYPE_SOLUTION"]
    N_NODES = DICTIONARY["N_NODES"]
    N_MATERIALS = DICTIONARY["N_MATERIALS"]
    N_SECTIONS = DICTIONARY["N_SECTIONS"]
    N_ELEMENTS = DICTIONARY["N_ELEMENTS"]
    N_DOFPRESCRIPTIONS = DICTIONARY["N_DISPLACEMENTS"]
    N_ELEMENTSLOADED = DICTIONARY["N_ELEMENTSLOADED"]
    N_NODESLOADED = DICTIONARY["N_NODESLOADED"]
    N_SPRINGS = DICTIONARY["N_SPRINGS"]
    COORDINATES = DICTIONARY["COORDINATES"]
    ELEMENTS = DICTIONARY["ELEMENTS"]
    MATERIALS = DICTIONARY["MATERIALS"]
    SECTIONS = DICTIONARY["SECTIONS"]
    PRESCRIPTIONS = DICTIONARY["PRESCRIBED DISPLACEMENTS"]
    ELEMENT_EXTERNAL_LOAD = DICTIONARY["ELEMENT LOADS"]
    NODAL_EXTERNAL_LOAD = DICTIONARY["NODAL LOADS"]
    SPRINGS = DICTIONARY["SPRINGS"]
    return TYPE_SOLUTION, TYPE_ELEMENT, N_NODES, N_MATERIALS, N_SECTIONS, N_ELEMENTS, N_DOFPRESCRIPTIONS, N_ELEMENTSLOADED, N_NODESLOADED, N_SPRINGS, COORDINATES, ELEMENTS, MATERIALS, SECTIONS, PRESCRIPTIONS, ELEMENT_EXTERNAL_LOAD, NODAL_EXTERNAL_LOAD, SPRINGS

def GET_VALUE_FROM_TXT_MEF1D_FINITO(FILENAME):
    """ THIS FUNCTION READ INPUT FILE """
    # OPEN FILE
    FILE = open(FILENAME, "r")
    # DATASET READ
    DATASET = FILE.read().split("\n")
    # STRUCTURE QUANTITIES
    TYPE_ELEMENT = int(DATASET.pop(0).split(':')[1])
    TYPE_SOLUTION = int(DATASET.pop(0).split(':')[1])
    N_NODES = int(DATASET.pop(0).split(':')[1])
    N_MATERIALS = int(DATASET.pop(0).split(':')[1])
    N_SECTIONS = int(DATASET.pop(0).split(':')[1])
    N_ELEMENTS = int(DATASET.pop(0).split(':')[1])
    N_DOFPRESCRIPTIONS = int(DATASET.pop(0).split(':')[1])
    N_ELEMENTSLOADED = int(DATASET.pop(0).split(':')[1])
    N_NODESLOADED = int(DATASET.pop(0).split(':')[1]) 
    N_SPRINGS = int(DATASET.pop(0).split(':')[1]) 
    # ELIMINATING SPACES AND TITLE
    DATASET.pop(0)
    DATASET.pop(0)
    # READ COORDINATES
    COORDINATES = np.zeros((N_NODES, 2))
    for I_COUNT in range(N_NODES):
        VALUES = DATASET.pop(0).split(',')
        COORDINATES[int(VALUES[0]),0] = float(VALUES[1])
        COORDINATES[int(VALUES[0]),1] = float(VALUES[2])
    # ELIMINATING SPACES AND TITLE
    DATASET.pop(0)
    DATASET.pop(0)
    # READ ELEMENTS
    ELEMENTS = np.zeros((N_ELEMENTS, 6))
    for J_COUNT in range(N_ELEMENTS):
        VALUES = DATASET.pop(0).split(',')
        ELEMENTS[int(VALUES[0]),0] = int(VALUES[1])
        ELEMENTS[int(VALUES[0]),1] = int(VALUES[2])    
        ELEMENTS[int(VALUES[0]),2] = int(VALUES[3])
        ELEMENTS[int(VALUES[0]),3] = int(VALUES[4])
        ELEMENTS[int(VALUES[0]),4] = int(VALUES[5])
        ELEMENTS[int(VALUES[0]),5] = int(VALUES[6])
    # ELIMINATING SPACES AND TITLE
    DATASET.pop(0)
    DATASET.pop(0)
    # READ MATERIALS
    MATERIALS = np.zeros((N_MATERIALS, 4))
    for K_COUNT in range(N_MATERIALS):
        VALUES = DATASET.pop(0).split(',')
        MATERIALS[int(VALUES[0]),0] = float(VALUES[1])
        MATERIALS[int(VALUES[0]),1] = float(VALUES[2])    
        MATERIALS[int(VALUES[0]),2] = float(VALUES[3])
        MATERIALS[int(VALUES[0]),3] = float(VALUES[4])
    # ELIMINATING SPACES AND TITLE
    DATASET.pop(0)
    DATASET.pop(0)
    # READ SECTIONS
    SECTIONS = np.zeros((N_SECTIONS, 5))
    for L_COUNT in range(N_SECTIONS):
        VALUES = DATASET.pop(0).split(',')
        SECTIONS[int(VALUES[0]),0] = float(VALUES[1])
        SECTIONS[int(VALUES[0]),1] = float(VALUES[2])    
        SECTIONS[int(VALUES[0]),2] = float(VALUES[3])
        SECTIONS[int(VALUES[0]),3] = float(VALUES[4])
        SECTIONS[int(VALUES[0]),4] = float(VALUES[5])
    # ELIMINATING SPACES AND TITLE
    DATASET.pop(0)
    DATASET.pop(0)
    # READ PRESCRIBED DISPLACEMENTS
    PRESCRIPTIONS = np.zeros((N_DOFPRESCRIPTIONS, 3))
    for M_COUNT in range(N_DOFPRESCRIPTIONS):
        VALUES = DATASET.pop(0).split(',')
        PRESCRIPTIONS[int(VALUES[0]),0] = int(VALUES[1])
        PRESCRIPTIONS[int(VALUES[0]),1] = int(VALUES[2])
        PRESCRIPTIONS[int(VALUES[0]),2] = float(VALUES[3]) 
    # ELIMINATING SPACES AND TITLE
    DATASET.pop(0)
    DATASET.pop(0)
    # READ ELEMENT LOAD ("AINDA É NECESSÁRRIO IMPLEMENTAR VAMOS SEMPRE DEIXAR A LEITURA NULL")
    if N_ELEMENTSLOADED == 0:
        DATASET.pop(0)
        ELEMENT_EXTERNAL_LOAD = "null"
    else:
        ELEMENT_EXTERNAL_LOAD = np.zeros((N_ELEMENTSLOADED, 5))
        for N_COUNT in range(N_ELEMENTSLOADED):
            VALUES = DATASET.pop(0).split(',')
            ELEMENT_EXTERNAL_LOAD[int(VALUES[0]),0] = float(VALUES[1])
            ELEMENT_EXTERNAL_LOAD[int(VALUES[0]),1] = float(VALUES[2])    
            ELEMENT_EXTERNAL_LOAD[int(VALUES[0]),2] = float(VALUES[3])
            ELEMENT_EXTERNAL_LOAD[int(VALUES[0]),3] = float(VALUES[4])
    # ELIMINATING SPACES AND TITLE
    DATASET.pop(0)
    DATASET.pop(0)
    # READ NODAL LOAD
    NODAL_EXTERNAL_LOAD = np.zeros((N_NODESLOADED, 3))
    for O_COUNT in range(N_NODESLOADED):
        VALUES = DATASET.pop(0).split(',')
        NODAL_EXTERNAL_LOAD[int(VALUES[0]),0] = float(VALUES[1])
        NODAL_EXTERNAL_LOAD[int(VALUES[0]),1] = float(VALUES[2]) 
        NODAL_EXTERNAL_LOAD[int(VALUES[0]),2] = float(VALUES[3]) 
    # ELIMINATING SPACES AND TITLE
    DATASET.pop(0)
    DATASET.pop(0)
    # READ SPRING ELEMENTS
    if N_SPRINGS == 0:
        DATASET.pop(0)
        SPRINGS = "null"
    else:
        SPRINGS = np.zeros((N_SPRINGS, 3))
        for P_COUNT in range(N_SPRINGS):
            VALUES = DATASET.pop(0).split(',')
            SPRINGS[int(VALUES[0]),0] = int(VALUES[1])
            SPRINGS[int(VALUES[0]),1] = int(VALUES[2])
            SPRINGS[int(VALUES[0]),2] = float(VALUES[3]) 
    return TYPE_SOLUTION, TYPE_ELEMENT, N_NODES, N_MATERIALS, N_SECTIONS, N_ELEMENTS, N_DOFPRESCRIPTIONS, N_ELEMENTSLOADED, N_NODESLOADED, N_SPRINGS, COORDINATES, ELEMENTS, MATERIALS, SECTIONS, PRESCRIPTIONS, ELEMENT_EXTERNAL_LOAD, NODAL_EXTERNAL_LOAD, SPRINGS

def GET_VALUE_FROM_TXT_MEF2D_FINITO(FILENAME):
    """
    This function reads the modified input file from 
    the ACADMESH2D software (https://set.eesc.usp.br/?page_id=237)

    Input:
    FILENAME: Structural dataset (.txt extension)

    Output: 
    N_NODES: Number of nodes (integer);
    N_MATERIALS: Number of materials (integer);
    N_THICKNESS: Number of thicness (integer);
    N_ELEMENTST3: Number of CST element (integer);
    N_ELEMENTST6: Number of LST element (integer);    
    N_ELEMENTST10: Number of QST element (integer);
    N_FORCES: Number of nodal forces (integer);
    N_PRESSURES: Number of element pressures (integer);
    N_DISPLACEMENTS: Number of nodal displacement control (integer);
    TYPE_PLANE: Type of analysis in the plan (string);
                    EPT - Plane Stress
                    EPD - Plane Strain
    TYPE_ELEMENT: Type element in Finito algorithm (integer); 
                    0 - Frame bar element
                    1 - CST surface element
    TYPE_SOLUTION: Solution of the system of equations (integer);
                    0 - Condense procedure
                    1 - 0 and 1 algorithm
    TYPE_INTEGRATION: Type numerical integration (string);
    COORDINATES: Coordinates properties (Python Numpy array);
                    ID, X, Y
    MATERIALS: Materials properties (Python Numpy array);
                    ID, YOUNG, POISSON, DENSITY
    THICKNESS: Thickness properties (Python Numpy array);
                    ID, THICKNESS
    ELEMENTS: Elements properties (Python Numpy array);
                    ID, NODE 0 ... NODE N, MATERIAL ID, THICKNESS ID
    NODAL_EXTERNAL_LOAD: Nodal force properties (Python Numpy array);
                    ID, NODE ID, FX VALUE, FY VALUE 
    # # # # # # # # # # # # # # # # # # # # # # # # #

    PRESSURES: Under development 
    
    # # # # # # # # # # # # # # # # # # # # # # # # #
    PRESCRIPTIONS: Displacement properties (Python Numpy array);
                    ID, NODE ID, DIRECTION ('X', Y' and 'BOTH'), DISPLACMENT VALUE 
    """
    # Read input file general properties
    FILE = open(FILENAME, "r")
    DATASET = FILE.read().split("\n")
    N_NODES = int(DATASET.pop(0).split(':')[1])
    N_MATERIALS = int(DATASET.pop(0).split(':')[1])
    N_THICKNESS = int(DATASET.pop(0).split(':')[1])
    N_ELEMENTST3 = int(DATASET.pop(0).split(':')[1])
    N_ELEMENTST6 = int(DATASET.pop(0).split(':')[1])
    N_ELEMENTST10 = int(DATASET.pop(0).split(':')[1])
    N_FORCES = int(DATASET.pop(0).split(':')[1])
    N_PRESSURES = int(DATASET.pop(0).split(':')[1])
    N_DISPLACEMENTS = int(DATASET.pop(0).split(':')[1])
    TYPE_PLANE = DATASET.pop(0).split(':')[1]
    TYPE_ELEMENT = int(DATASET.pop(0).split(':')[1])
    TYPE_SOLUTION = int(DATASET.pop(0).split(':')[1])
    TYPE_INTEGRATION = int(DATASET.pop(0).split(':')[1])
    # Read coordinates 
    DATASET.pop(0)
    DATASET.pop(0)   
    COORDINATES = np.zeros((N_NODES, 3))
    for I_COUNT in range(N_NODES):
        VALUES = DATASET.pop(0).split(',')
        COORDINATES[int(VALUES[0]), 0] = float(VALUES[0])
        COORDINATES[int(VALUES[0]), 1] = float(VALUES[1])
        COORDINATES[int(VALUES[0]), 2] = float(VALUES[2])
    # Read materials
    DATASET.pop(0)
    DATASET.pop(0)
    MATERIALS = np.zeros((N_MATERIALS, 3))
    for J_COUNT in range(N_MATERIALS):
        VALUES1 = DATASET.pop(0).split(',')
        MATERIALS[int(VALUES1[0]), 0] = float(VALUES1[1])
        MATERIALS[int(VALUES1[0]), 1] = float(VALUES1[2])
    # Read thickness
    DATASET.pop(0)
    DATASET.pop(0)
    THICKNESS = np.zeros((N_THICKNESS, 1))
    for K_COUNT in range(N_THICKNESS):
        VALUES2 = DATASET.pop(0).split(',')
        THICKNESS[int(VALUES2[0]), 0] = float(VALUES2[1])
    # Read Elements
    DATASET.pop(0)
    DATASET.pop(0)
    # COLOCAR AQUI UM IF PARA DIZER QUE ESSA LEITURA É SÓ DO T3
    ELEMENTS = np.zeros((N_ELEMENTST3, 6))
    for L_COUNT in range(N_ELEMENTST3):
        VALUES3 = DATASET.pop(0).split(',')
        ELEMENTS[int(VALUES3[0]), 0] = int(VALUES3[0])
        ELEMENTS[int(VALUES3[0]), 1] = int(VALUES3[1])
        ELEMENTS[int(VALUES3[0]), 2] = int(VALUES3[2])
        ELEMENTS[int(VALUES3[0]), 3] = int(VALUES3[3])
        ELEMENTS[int(VALUES3[0]), 4] = int(VALUES3[4])
        ELEMENTS[int(VALUES3[0]), 5] = int(VALUES3[5])
    # Read nodal forces
    DATASET.pop(0)
    DATASET.pop(0)
    NODAL_EXTERNAL_LOAD = np.zeros((N_FORCES, 3))
    for M_COUNT in range(N_FORCES):
        VALUES4 = DATASET.pop(0).split(',')
        NODAL_EXTERNAL_LOAD[int(VALUES4[0]), 0] = float(VALUES4[1])  # Nó
        if float(VALUES4[2]) != 0:
            NODAL_EXTERNAL_LOAD[int(VALUES4[0]), 2] = float(VALUES4[2])
            NODAL_EXTERNAL_LOAD[int(VALUES4[0]), 1] = 0
        elif float(VALUES4[3]) != 0:
            NODAL_EXTERNAL_LOAD[int(VALUES4[0]), 2] = float(VALUES4[3])
            NODAL_EXTERNAL_LOAD[int(VALUES4[0]), 1] = 1
    # # # # # # # # # # # # # # # # # # # # # # # # #
    
    # Under development Pressure read
    
    # # # # # # # # # # # # # # # # # # # # # # # # # 
    # Read displacement
    DATASET.pop(0)
    DATASET.pop(0)
    COUNT = 0
    for K_COUNT in range(N_DISPLACEMENTS):
        VALUES5 = DATASET[K_COUNT].split(',')

        if VALUES5[2] == "'BOTH'":
            COUNT += 2

        else:
            COUNT += 1
    PRESCRIPTIONS = np.zeros((COUNT, 3))
    COUNT = 0
    for w in range(N_DISPLACEMENTS):
        VALUES6 = DATASET.pop(0).split(',')
        if VALUES6[2] == "'X'":
            PRESCRIPTIONS[COUNT, 0] = int(VALUES6[1])
            PRESCRIPTIONS[COUNT, 1] = 0
            PRESCRIPTIONS[COUNT, 2] = float(VALUES6[3])
            COUNT += 1
        elif VALUES6[2] == "'Y'":
            PRESCRIPTIONS[COUNT, 0] = int(VALUES6[1])
            PRESCRIPTIONS[COUNT, 1] = 1
            PRESCRIPTIONS[COUNT, 2] = float(VALUES6[3])
            COUNT += 1
        else:
            PRESCRIPTIONS[COUNT, 0] = int(VALUES6[1])
            PRESCRIPTIONS[COUNT, 1] = 0
            PRESCRIPTIONS[COUNT, 2] = float(VALUES6[3])
            COUNT += 1
            PRESCRIPTIONS[COUNT, 0] = int(VALUES6[1])
            PRESCRIPTIONS[COUNT, 1] = 1
            PRESCRIPTIONS[COUNT, 2] = float(VALUES6[3])
            COUNT += 1
    return N_NODES, N_MATERIALS, N_THICKNESS, N_ELEMENTST3, N_ELEMENTST6, N_ELEMENTST10, N_FORCES, N_PRESSURES, N_DISPLACEMENTS, TYPE_PLANE, TYPE_ELEMENT, TYPE_SOLUTION, TYPE_INTEGRATION, COORDINATES, MATERIALS, THICKNESS, ELEMENTS, NODAL_EXTERNAL_LOAD, PRESCRIPTIONS

def INDEX_ASSEMBLY(TYPE_ELEMENT):
    """ 
    This function establishes values for the internal variables 
    that are used in the algorithm
    
    Input:
    TYPE_ELEMENT: Type element in Finito algorithm (integer);
                    0 - Frame bar element
                    1 - CST surface element

    Output:
    N_DOFSNODE: Number of degress of freedom per node (integer);
    N_NODESELEMENT: Number of nodes per element (integer);
    DOFS_ACTIVE: Auxiliary variable (Python list);
    DOFS_LOCAL: ID local degrees of freedom in element (Python list);
    AUX_1: Material ID in ELEMENTS (integer);
    AUX_2: Geometry ID in ELEMENTS (integer);
    N_DOFSELEMENT: Number of degress of freedom per element (integer);
    """
    # Frame bar element
    if TYPE_ELEMENT == 0: 
        N_DOFSNODE = 3
        N_NODESELEMENT = 2
        DOFS_ACTIVE = [1, 1, 1]
        DOFS_LOCAL = [0, 1, 2, 3, 4, 5]
    # CST surface element
    elif TYPE_ELEMENT == 1:
        N_DOFSNODE = 2
        N_NODESELEMENT = 3
        DOFS_ACTIVE = [1, 1, 1]
        DOFS_LOCAL = [0, 1, 2, 3, 4, 5]
    AUX_1 = N_NODESELEMENT + 0
    AUX_2 = N_NODESELEMENT + 1
    N_DOFSELEMENT = N_DOFSNODE * N_NODESELEMENT
    return N_DOFSNODE, N_NODESELEMENT, DOFS_ACTIVE, DOFS_LOCAL, AUX_1, AUX_2, N_DOFSELEMENT

def DOF_GLOBAL_ASSEMBLY(TYPE_ELEMENT, N_DOFSNODE, N_NODES):
    """
    This function determines the value of the degrees of global 
    freedoms by node of the structure
    
    Input
    TYPE_ELEMENT: Type element in Finito algorithm (integer); 
                    0 - Frame bar element
                    1 - CST surface element
    N_DOFSNODE: Number of degress of freedom per node (integer);
    N_NODES: Number of nodes (integer);

    Output:
    DOF_GLOBALNODAL: ID global DOF per node (Python Numpy array);
    """
    DOF_GLOBALNODAL = np.zeros((N_NODES, N_DOFSNODE))
    for I_COUNT in range (N_NODES):
        # Frame bar element
        if TYPE_ELEMENT == 0: 
            DOF_GLOBALNODAL[I_COUNT, 0] = int(N_DOFSNODE * I_COUNT + 0)
            DOF_GLOBALNODAL[I_COUNT, 1] = int(N_DOFSNODE * I_COUNT + 1)
            DOF_GLOBALNODAL[I_COUNT, 2] = int(N_DOFSNODE * I_COUNT + 2)
        # CST surface element
        elif TYPE_ELEMENT == 1:
            DOF_GLOBALNODAL[I_COUNT, 0] = int(N_DOFSNODE * I_COUNT + 0)
            DOF_GLOBALNODAL[I_COUNT, 1] = int(N_DOFSNODE * I_COUNT + 1)
    return DOF_GLOBALNODAL

def TOTAL_DEGREE_FREEDOM(N_DOFSNODE, N_NODES):
    """ 
    This function determines the quantity and values of the 
    structure's global degrees of freedom

    Input:
    N_DOFSNODE: Number of degress of freedom per node (integer);
    N_NODES: Number of nodes (integer);

    Output:
    DOF_GLOBAL: ID global degree of freedom (Python list); 
    N_DOFSGLOBAL: Total of degrees of freedom (integer);
    """
    DOF_GLOBAL = []
    N_DOFSGLOBAL = N_NODES * N_DOFSNODE
    for I_COUNT in range (N_DOFSGLOBAL):
        DOF_GLOBAL.append(I_COUNT)
    return DOF_GLOBAL, N_DOFSGLOBAL

def PRESCRIPTIONS_DEGREE_FREEDOM(PRESCRIPTIONS, DOF_GLOBALNODAL, N_DOFSNODE):
    """
    This function determines the quantity and values of the displacement 
    prescriptions of degrees of freedom 

    Input: 
    DOF_GLOBALNODAL: ID global DOF per node (Python Numpy array);
    PRESCRIPTIONS: Displacement properties (Python Numpy array);
                ID, NODE ID, DIRECTION ('X', Y' and 'BOTH'), DISPLACMENT VALUE
    N_DOFSNODE: Number of degress of freedom per node (integer);

    Output:
    DOF_PRESCRIPTIONS: ID prescribed degree of freedom (Python list); 
    DOF_PRESCRIPTIONSVALUE: Value prescribed degree of freedom (Python list);
    N_DOFSPRESCRIPTIONS: Total number of prescribed degrees of freedom (integer);
    """
    DOF_PRESCRIPTIONS = []
    DOF_PRESCRIPTIONSVALUE = []
    N_DOFSPRESCRIPTIONS = PRESCRIPTIONS.shape[0]
    for I_COUNT in range (N_DOFSPRESCRIPTIONS):
        NODE = int(PRESCRIPTIONS[I_COUNT, 0])
        INDEX_DOF = int(PRESCRIPTIONS[I_COUNT, 1])
        DOF_VALUE = int(DOF_GLOBALNODAL[NODE, INDEX_DOF])
        DOF_PRESCRIPTIONS.append(DOF_VALUE)
        PRESCRIBED_VALUE = PRESCRIPTIONS[I_COUNT, 2]
        DOF_PRESCRIPTIONSVALUE.append(PRESCRIBED_VALUE)
    return DOF_PRESCRIPTIONS, DOF_PRESCRIPTIONSVALUE, N_DOFSPRESCRIPTIONS

def FREE_DEGREE_FREEDOM(DOF_PRESCRIPTIONS, DOF_GLOBAL):
    """
    This function determines the quantity and values of the displacement 
    free of degrees of freedom 

    Input:
    DOF_PRESCRIPTIONS: ID prescribed degree of freedom (Python list); 
    DOF_GLOBAL: ID global degree of freedom (Python list);

    Output:
    DOF_FREE: ID free degree of freedom (Python list); 
    N_DOFSFREE: Total number of free degrees of freedom (integer);
    """
    DOF_FREE = np.setdiff1d(DOF_GLOBAL, DOF_PRESCRIPTIONS)
    N_DOFSFREE = len(DOF_FREE)
    return DOF_FREE, N_DOFSFREE

def CONTRIBUTION_NODAL_EXTERNAL_LOAD(NODAL_EXTERNAL_LOAD, N_DOFSGLOBAL, DOF_GLOBALNODAL):
    """ THIS FUNCTION BUILDS THE VECTOR OF EXTERNAL LOADS ON THE LEVEL OF FREEDOM. 
    IN THIS CASE THIS FUNCTION REPRESENTS THE CONTRIBUTION OF NODAL LOADS """
    DOF_NODALFORCE = np.zeros((N_DOFSGLOBAL, 1))
    N_NODALFORCE = NODAL_EXTERNAL_LOAD.shape[0]
    for I_COUNT in range(N_NODALFORCE):
        NODE = int(NODAL_EXTERNAL_LOAD[I_COUNT, 0])
        INDEX_DOF = int(NODAL_EXTERNAL_LOAD[I_COUNT, 1])
        DOF_VALUE = int(DOF_GLOBALNODAL[NODE, INDEX_DOF])
        P = int(NODAL_EXTERNAL_LOAD[I_COUNT, 2])
        DOF_NODALFORCE[DOF_VALUE, 0] = P
    return DOF_NODALFORCE

def MATERIALS_PROPERTIES_0(ELEMENTS, MATERIALS, I_ELEMENT, AUX_1):
    """
    THIS FUNCTION CREATES A VECTOR WITH THE MATERIAL INFORMATION 
    OF THE I_ELEMENT ELEMENT
    """
    MATERIAL_ID = int(ELEMENTS[I_ELEMENT, AUX_1])
    E = MATERIALS[MATERIAL_ID, 0]
    NU = MATERIALS[MATERIAL_ID, 1]
    PHO = MATERIALS[MATERIAL_ID, 2]
    ALPHA = MATERIALS[MATERIAL_ID, 3]
    G = E / (2 * (1 + NU))
    MATERIAL_IELEMENT = [E, G, NU, ALPHA, PHO]
    return MATERIAL_IELEMENT

def GEOMETRIC_PROPERTIES_0(COORDINATES, ELEMENTS, SECTIONS, I_ELEMENT, AUX_2):
    """ 
    This function assigns the bar element's geometric propertiest
    """
    NODE_1 = int(ELEMENTS[I_ELEMENT, 0])
    NODE_2 = int(ELEMENTS[I_ELEMENT, 1])
    X_NODE1 = COORDINATES[NODE_1, 0]
    Y_NODE1 = COORDINATES[NODE_1, 1]
    X_NODE2 = COORDINATES[NODE_2, 0]
    Y_NODE2 = COORDINATES[NODE_2, 1]
    DELTA_X = X_NODE2 - X_NODE1
    DELTA_Y = Y_NODE2 - Y_NODE1
    L = ((DELTA_X) ** 2 + (DELTA_Y) ** 2) ** 0.50
    COS = DELTA_X / L
    SIN = DELTA_Y / L
    SECTION_ID = int(ELEMENTS[I_ELEMENT, AUX_2])
    A = SECTIONS[SECTION_ID, 0]
    I_1 = SECTIONS[SECTION_ID, 1]
    I_2 = SECTIONS[SECTION_ID, 2]
    SECTION_IELEMENT = [L, SIN, COS, A, I_1, I_2]
    return SECTION_IELEMENT

def GEOMETRIC_PROPERTIES_1(COORDINATES, ELEMENTS, I_ELEMENT, THICKNESS, AUX_2):
    """ 
    This function assigns the surface element's geometric propertiest
    
    Input:
    COORDINATES: Coordinates properties (Python Numpy array);
                    ID, X, Y
    THICKNESS: Thickness properties (Python Numpy array);
                    ID, THICKNESS
    I_ELEMENT: ID i element in looping for
                    0 and (N_ELEMENTS - 1) 
    ELEMENTS: Elements properties (Python Numpy array);
                    ID, NODE 0 ... NODE N, MATERIAL ID, THICKNESS ID
    Output: 
    SECTION_IELEMENT: i element geometric properties (Python dictionary);
    """
    NODE_0 = int(ELEMENTS[I_ELEMENT, 1])
    X0, Y0 = COORDINATES[NODE_0, 1], COORDINATES[NODE_0, 2]
    NODE_1 = int(ELEMENTS[I_ELEMENT, 2])
    X1, Y1 = COORDINATES[NODE_1, 1], COORDINATES[NODE_1, 2]
    NODE_2 = int(ELEMENTS[I_ELEMENT, 3])
    X2, Y2 = COORDINATES[NODE_2, 1], COORDINATES[NODE_2, 2]
    THICK_ID = int(ELEMENTS[I_ELEMENT, AUX_2])
    THICK = THICKNESS[THICK_ID, 0]
    SECTION_IELEMENT = {'X_E': np.array([[X0, Y0], [X1, Y1], [X2, Y2]]), 'THICKNESS': THICK}
    return SECTION_IELEMENT

def CONSTITUTIVE_C(TYPE_PLANE, MATERIALS, ELEMENTS, I_ELEMENT):
    """
    This function determines the matrix responsible for establishing 
    the constitutive relationship between stress and strain 
    according to the choosen analysis
    
    Input:
    TYPE_PLANE: Type of analysis in the plan (string);
                    EPT - Plane Stress
                    EPD - Plane Strain
    MATERIALS: Materials properties (Python Numpy array);
                    ID, YOUNG, POISSON, DENSITY
    ELEMENTS: Elements properties (Python Numpy array);
                    ID, NODE 0 ... NODE N, MATERIAL ID, THICKNESS ID
    I_ELEMENT: ID i element in looping for
                    0 and (N_ELEMENTS - 1) 
    
    Output:
    C: Constitutive matrix in formulation (Python Numpy array); 
    """
    MATERIAL_ID = int(ELEMENTS[I_ELEMENT, 4])
    E = MATERIALS[MATERIAL_ID, 0]
    NU = MATERIALS[MATERIAL_ID, 1]
    # Plane stress
    if TYPE_PLANE == 'EPT':
            C11 = 1
            C12 = NU
            C21 = C12
            C22 = 1
            C33 = 0.5 * (1 - NU)
            AUX_1 = E / (1 - NU ** 2)
            AUX_2 = np.array([[C11, C12, 0], [C21, C22, 0], [0, 0, C33]])
            C = AUX_1 * AUX_2
    # Plane strain
    elif TYPE_PLANE == 'EPD':
            C11 = 1 - NU
            C12 = NU
            C21 = NU
            C22 = 1 - NU
            C33 = 0.5 - NU
            AUX_1 = E/((1 + NU)*(1 - 2*NU))
            AUX_2 = np.array([[C11, C12, 0],[C21, C22, 0],[0, 0, C33]])
            C = AUX_1 * AUX_2
    return C

def HINGED_PROPERTIES(ELEMENTS):
    HINGES = ELEMENTS[:,4:]
    return HINGES

def ELEMENT_STIFFNESS_0(TYPE_ELEMENT, SECTION_IELEMENT, MATERIAL_IELEMENT, HINGES_IELEMENT):
    """ THIS FUNCTION CREATES THE ELEMENT STIFFNESS I_ELEMENT """
    # http://www.ikb.poznan.pl/przemyslaw.litewka/06-matrix-stiffness-method.pdf
    if (TYPE_ELEMENT == 0 and HINGES_IELEMENT[0] == 0 and HINGES_IELEMENT[1] == 0):
        L = SECTION_IELEMENT[0]
        A = SECTION_IELEMENT[3]
        I = SECTION_IELEMENT[5]
        E = MATERIAL_IELEMENT[0]
        C1 = A * E / L
        C2 = E * I / (L ** 3)
        K_IELEMENT = np.array([[C1, 0, 0, -C1, 0, 0],
                               [0, 12 * C2, 6 * C2 * L, 0, -12 * C2, 6 * C2 * L],
                               [0, 6 * C2 * L, 4 * C2 * L ** 2, 0, -6 * C2 * L, 2 * C2 * L ** 2],
                               [-C1, 0, 0, C1, 0, 0],
                               [0, -12 * C2, -6 * C2 * L, 0, 12 * C2, -6 * C2 * L],
                               [0, 6 * C2 * L, 2 * C2 * L ** 2, 0, -6 * C2 * L, 4 * C2 * L **2]])
    elif (TYPE_ELEMENT == 0 and HINGES_IELEMENT[0] == 0 and HINGES_IELEMENT[1] == 1):
        L = SECTION_IELEMENT[0]
        A = SECTION_IELEMENT[3]
        I = SECTION_IELEMENT[5]
        E = MATERIAL_IELEMENT[0]
        C1 = A * E / L
        C2 = E * I / (L ** 3)
        K_IELEMENT = np.array([[C1, 0, 0, -C1, 0, 0],
                               [0, 3 * C2, 3 * C2 * L, 0, -3 * C2, 0],
                               [0, 3 * C2 * L, 3 * C2 * L ** 2, 0, -3 * C2 * L, 0],
                               [-C1, 0, 0, C1, 0, 0],
                               [0, -3 * C2, -6 * C2 * L, 0, 3 * C2, 0],
                               [0, 0, 0, 0, 0, 0]])    
    elif (TYPE_ELEMENT == 0 and HINGES_IELEMENT[0] == 1 and HINGES_IELEMENT[1] == 0):
        L = SECTION_IELEMENT[0]
        A = SECTION_IELEMENT[3]
        I = SECTION_IELEMENT[5]
        E = MATERIAL_IELEMENT[0]
        C1 = A * E / L
        C2 = E * I / (L ** 3)
        K_IELEMENT = np.array([[C1, 0, 0, -C1, 0, 0],
                               [0, 3 * C2, 0, 0, -3 * C2, 3 * C2 * L],
                               [0, 0, 0, 0, 0, 0],
                               [-C1, 0, 0, C1, 0, 0],
                               [0, -3 * C2, 0, 0, 3 * C2, -3 * C2 * L],
                               [0, 3 * C2 * L, 0, 0, -3 * C2 * L, 3 * C2 * L **2]])     

    elif (TYPE_ELEMENT == 0 and HINGES_IELEMENT[0] == 1 and HINGES_IELEMENT[1] == 1):
        L = SECTION_IELEMENT[0]
        A = SECTION_IELEMENT[3]
        I = SECTION_IELEMENT[5]
        E = MATERIAL_IELEMENT[0]
        C1 = A * E / L
        C2 = E * I / (L ** 3)
        K_IELEMENT = np.array([[C1, 0, 0, -C1, 0, 0],
                              [0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0],
                              [-C1, 0, 0, C1, 0, 0],
                              [0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0]])
    return K_IELEMENT

def SHAPE_FUNCTIONS(TYPE_ELEMENT, N_NODESELEMENT, ISO_COORDINATES):
    """
    This function creates the matrix of the derivatives of the shape functions

    Input:
    TYPE_ELEMENT: Type element in Finito algorithm (integer); 
                0 - Frame bar element
                1 - CST surface element
    N_NODESELEMENT: Number of nodes per element (integer);
    ISO_COORDINATES: Isoparametric coordinates (Python dictionary);

    Output:
    ND_DIFF: ND derivatives matrix (Python Numpy array);
    NX_DIFF: NX derivatives matrix (Python Numpy array);
    """
    if TYPE_ELEMENT == 1:
        # Derivative shape functions 
        DIFF_KSI = [-1, 1, 0]
        DIFF_ETA = [-1, 0, 1]
        # ND e NX assembly
        NX_DIFF = np.array([DIFF_KSI, DIFF_ETA])
        ND_DIFF = DND_ASSEMBLY(N_NODESELEMENT, NX_DIFF)
    return ND_DIFF, NX_DIFF

def DND_ASSEMBLY(N_NODESELEMENT, NX_DIFF):
    """
    This function assembles the derived matrix ND

    Input:
    N_NODESELEMENT: Number of nodes per element (integer);
    NX_DIFF: NX derivatives matrix (Python Numpy array);
    
    Output:
    ND_DIFF: ND derivatives matrix (Python Numpy array);
    """
    ND_DIFF_1 = np.zeros((2, 2 * N_NODESELEMENT))
    ND_DIFF_2 = np.zeros((2, 2 * N_NODESELEMENT))
    # Automatic assembly 
    for I_COUNT in range(2):
        COUNT_1 = 0
        COUNT_2 = 0
        for J_COUNT in range(0, 2 * N_NODESELEMENT, 2):
            ND_DIFF_1[I_COUNT, J_COUNT] = NX_DIFF[I_COUNT, COUNT]
            COUNT_1 += 1
        for K_COUNT in range(1, 2 * N_NODESELEMENT, 2):
            ND_DIFF_2[I_COUNT, K_COUNT] = NX_DIFF[I_COUNT, COUNT]
            COUNT_2 += 1
    ND_DIFF = np.vstack((ND_DIFF_1, ND_DIFF_2))
    return ND_DIFF

def STIFFNESS(NX_DIFF, ND_DIFF, C, X_E):
    """
    This function assembles the element's stiffness matrix to a Gaussian point

    Input:
    NX_DIFF: NX derivatives matrix (Python Numpy array);
    ND_DIFF: ND derivatives matrix (Python Numpy array);
    C: Constitutive matrix in formulation (Python Numpy array); 
    X_E: i element coordinates (Python Numpy array);

    Output:
    K_I: i element stiffness matrix (Python Numpy array);
    """
    # Jacobian matrix
    J = np.dot(NX_DIFF, X_E)
    # Determinant of the Jacobian matrix
    DET_J = np.linalg.det(J)
    # Gamma and Gamma U matrix
    GAMMA = np.linalg.inv(J)
    GAMMA_00 = GAMMA[0, 0]
    GAMMA_01 = GAMMA[0, 1]
    GAMMA_10 = GAMMA[1, 0]
    GAMMA_11 = GAMMA[1, 1]
    GAMMA_U = np.array([[GAMMA_00, GAMMA_01, 0, 0],
                        [GAMMA_10, GAMMA_11, 0, 0],
                        [0, 0, GAMMA_00, GAMMA_01],
                        [0, 0, GAMMA_10, GAMMA_11]])
    H = np.array([[1, 0, 0, 0],
                    [0, 0, 0, 1],
                    [0, 1, 1, 0]])
    B = np.dot(np.dot(H, GAMMA_U), ND_DIFF)
    # i element stiffness matrix
    K_I = THICK * (np.dot(np.dot(B.transpose(), C), np.dot(B, DET_J)))
    return K_I

def NUMERICAL_INTEGRATION(TYPE_INTEGRATION):
    """
    This function creates the parameters for numerical integration
    
    Input:
    TYPE_INTEGRATION: Type numerical integration (string);

    Output:
    NUM_INT: Setup numerical integration (Python dictionary);
    """
    # Hammer 12 points
    if TYPE_INTEGRATION == 'HAMMER-12':
        W = [0.116786275726379/2.0, 0.116786275726379/2.0, 0.116786275726379/2.0,
                            0.050844906370207/2.0, 0.050844906370207/2.0, 0.050844906370207/2.0,
                            0.082851075618374/2.0, 0.082851075618374/2.0, 0.082851075618374/2.0,
                            0.082851075618374/2.0, 0.082851075618374/2.0, 0.082851075618374/2.0]
        KSI = [0.501426509658179, 0.249286745170910, 0.249286745170910,
                               0.873821971016996, 0.063089014491502, 0.063089014491502,
                               0.053145049844816, 0.310352451033785, 0.636502499121399,
                               0.310352451033785, 0.636502499121399, 0.053145049844816]
        ETA = [0.249286745170910, 0.249286745170910, 0.501426509658179,
                               0.063089014491502, 0.063089014491502, 0.873821971016996,
                               0.310352451033785, 0.636502499121399, 0.053145049844816,
                               0.053145049844816, 0.310352451033785, 0.636502499121399]
        N_POINTS = W.shape[1]
    NUM_INT = {'W': W, 'KSI': KSI, 'ETA': ETA, 'N': N_POINTS}
    return NUM_INT

def ELEMENT_STIFFNESS_1(NUM_INT, N_DOFSELEMENT, TYPE_ELEMENT, N_NODESELEMENT, C_IELEMENT, SECTION_IELEMENT):
    """
    This function calculates the stiffness matrix of the isoparametric element

    Input:
    NUM_INT: Setup numerical integration (Python dictionary);
    N_DOFSELEMENT: Number of degress of freedom per element (integer);
    TYPE_ELEMENT: Type element in Finito algorithm (integer); 
            0 - Frame bar element
            1 - CST surface element
    N_NODESELEMENT: Number of nodes per element (integer);
    C_IELEMENT: i element constitutive matrix in formulation (Python Numpy array); 
    SECTION_IELEMENT: i element geometric properties (Python dictionary);

    Output:
    K_IELEMENT: Complete i element stiffness matrix
    """
    POINTS = NUM_INT['N']
    K_IELEMENT = np.zeros((N_DOFSELEMENT, N_DOFSELEMENT))
    for I_COUNT in range(POINTS):
        ISO_COORDINATES = {'KSI': NUM_INT['KSI'][I_COUNT], 'ETA': NUM_INT['ETA'][I_COUNT]}
        [ND_DIFF, NX_DIFF] = SHAPE_FUNCTIONS(TYPE_ELEMENT, N_NODESELEMENT, ISO_COORDINATES):
        X_E = SECTION_IELEMENT['X_E']
        K_I = STIFFNESS(NX_DIFF, ND_DIFF, C_IELEMENT, X_E):
        WEIGHT = NUM_INT['W'][I_COUNT]
        K_IELEMENT += K_I * WEIGHT
    return K_IELEMENT

def SPRING_CONTRIBUTION(N_DOFSNODE, SPRINGS, N_SPRINGS):
    """ THIS FUNCTION CREATES THE ELEMENT ROTATION I_ELEMENT """
    SPRING_INDEX = []
    SPRING_VALUES = []
    for I_COUNT in range(N_SPRINGS):
        NODE = int(SPRINGS[I_COUNT, 0])
        INDEX_DOF = int(SPRINGS[I_COUNT, 1])
        SPRING_INDEX.append(int(N_DOFSNODE * NODE + INDEX_DOF))
        SPRING_VALUES.append(SPRINGS[I_COUNT, 2])
    return SPRING_INDEX, SPRING_VALUES

def ELEMENT_ROTATION(TYPE_ELEMENT, SECTION_IELEMENT):
    """ THIS FUNCTION CREATES THE ELEMENT ROTATION I_ELEMENT """
    SIN = SECTION_IELEMENT[1]
    COS = SECTION_IELEMENT[2]
    if TYPE_ELEMENT == 0:
        R_IELEMENT = np.array([[COS, SIN, 0, 0, 0, 0],
                               [-SIN, COS, 0, 0, 0, 0],
                               [0, 0, 1, 0, 0, 0],
                               [0, 0, 0, COS, SIN, 0],
                               [0, 0, 0, -SIN, COS, 0],
                               [0, 0, 0, 0, 0, 1]])
    return R_IELEMENT

# GlOBAL DOF I_ELEMENT
def GLOBAL_DOF_ELEMENT(N_NODESELEMENT, N_DOFSNODE, DOF_GLOBALNODAL, ELEMENTS, I_ELEMENT):
    """ THIS FUNCTION DETERMINES THE DEGREE OF GLOBAL FREEDOM OF THE ELEMENT I_ELEMENT """
    DOF_GLOBALIELEMENT = []
    for I_COUNT in range(N_NODESELEMENT):
        NODE = int(ELEMENTS[I_ELEMENT, I_COUNT])
        for J_COUNT in range(N_DOFSNODE):
            DOF_VALUE = int(DOF_GLOBALNODAL[NODE, J_COUNT])
            DOF_GLOBALIELEMENT.append(DOF_VALUE)
    return DOF_GLOBALIELEMENT

# GlOBAL STIFFNESS ASSEMBLY
def GLOBAL_STIFFNESS(N_DOFSGLOBAL, DOF_GLOBALIELEMENT, K_IELEMENT):
    """ THIS FUNCTION CREATES THE GLOBAL STIFFNESS """
    K_G = np.zeros((N_DOFSGLOBAL, N_DOFSGLOBAL))
    for I_COUNT, I_VALUE in enumerate(DOF_GLOBALIELEMENT):
        for J_COUNT, J_VALUE in enumerate(DOF_GLOBALIELEMENT):
            K_G[I_VALUE, J_VALUE] = K_G[I_VALUE, J_VALUE] + K_IELEMENT[I_COUNT, J_COUNT]
    return K_G

# CONDENSE GLOBAL FREE STIFFNESS 
def CONDENSE_GLOBAL_FREE_STIFFNESS(K_G, DOF_FREE, N_DOFSFREE):
    """ A FAZER """
    K_FF = np.zeros((N_DOFSFREE, N_DOFSFREE))
    for I_COUNT in range(N_DOFSFREE):
        DOF_FREELINE = DOF_FREE[I_COUNT]
        for J_COUNT in range(N_DOFSFREE):
            DOF_FREECOLUMN = DOF_FREE[J_COUNT]
            K_FF[I_COUNT, J_COUNT] = K_G[DOF_FREELINE, DOF_FREECOLUMN]
    return K_FF

# CONDENSE GLOBAL PRESCRIBED FREE STIFFNESS 
def CONDENSE_PRESCRIBED_FREE_GLOBAL_STIFFNESS(K_G, DOF_FREE, N_DOFSFREE, DOF_PRESCRIPTIONS, N_DOFSPRESCRIPTIONS):
    """ A FAZER """
    K_PF = np.zeros((N_DOFSPRESCRIPTIONS, N_DOFSFREE))
    for I_COUNT in range(N_DOFSPRESCRIPTIONS):
        DOF_PRECRIBEDLINE = DOF_PRESCRIPTIONS[I_COUNT]
        for J_COUNT in range(N_DOFSFREE):
            DOF_FREECOLUMN = DOF_FREE[J_COUNT]
            K_PF[I_COUNT, J_COUNT] = K_G[DOF_PRECRIBEDLINE, DOF_FREECOLUMN]
    return K_PF

# CONDENSE GLOBAL PRESCRIBED DISPLACEMENT
def CONDENSE_PRESCRIBED_GLOBAL_DISPLACEMENT(DOF_PRESCRIPTIONSVALUE, N_DOFSPRESCRIPTIONS):
    U_PP = np.zeros((N_DOFSPRESCRIPTIONS, 1))
    for I_COUNT in range(N_DOFSPRESCRIPTIONS):
        DOF_PRESCRIBEDVALUE = DOF_PRESCRIPTIONSVALUE[I_COUNT]
        U_PP[I_COUNT, 0] = DOF_PRESCRIBEDVALUE
    return U_PP

# CONDENSE GLOBAL FREE EXTERNAL FORCES
def CONDENSE_FREE_GLOBAL_FORCES(F_EXT, DOF_FREE, N_DOFSFREE):
    F_FF = np.zeros((N_DOFSFREE, 1))
    for I_COUNT in range(N_DOFSFREE):
        FREE_DOF = DOF_FREE[I_COUNT]
        F_FF[I_COUNT, 0] = F_EXT[FREE_DOF, 0]
    return F_FF

# ASSEMBLY GLOBAL DISPLACEMENT
def ASSEMBLY_TOTAL_DISPLACEMENT(U_FF, U_PP, N_DOFSGLOBAL, DOF_PRESCRIPTIONS, DOF_FREE):
    U_G = np.zeros((N_DOFSGLOBAL, 1))
    for I_COUNT, I_VALUE in enumerate(DOF_PRESCRIPTIONS):
        DOF_DISPLACEMENTVALUE = U_PP[I_COUNT, 0]
        U_G[I_VALUE, 0] = DOF_DISPLACEMENTVALUE
    for J_COUNT, J_VALUE in enumerate(DOF_FREE):
        DOF_DISPLACEMENTVALUE = U_FF[J_COUNT, 0]
        U_G[J_VALUE, 0] = DOF_DISPLACEMENTVALUE
    return U_G

# CONDENSE GLOBAL DISPLACEMENTS IN I_ELEMENT
def CONDENSE_GLOBAL_ELEMENT_DISPLACEMENTS(U_G, N_DOFSELEMENT, DOF_GLOBALIELEMENT):
    U_IELEMENT = np.zeros((N_DOFSELEMENT, 1))
    for I_COUNT, J_COUNT in enumerate(DOF_GLOBALIELEMENT):
        U_IELEMENT[I_COUNT, 0] = U_G[J_COUNT, 0]
    return U_IELEMENT

# ASSEMBLY INTERNAL LOADS
def GLOBAL_INTERNAL_LOADS(F_INTIELEMENT, N_DOFSGLOBAL, DOF_GLOBALIELEMENT):
    F_INT = np.zeros((N_DOFSGLOBAL, 1))
    for I_COUNT, J_COUNT in enumerate(DOF_GLOBALIELEMENT):
        F_INT[J_COUNT, 0] = F_INTIELEMENT[I_COUNT, 0]
    return F_INT