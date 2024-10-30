# Tutorial 10 - Bioinformatics

# DNA Sequencing

# Standard imports
import numpy as np

from enum import Enum
from blosum_62 import blosum_62

# Initialise Constants
basic_scoring_matrix = {
    'type' : 'basic',
    'match score' : 1,
    'mismatch score' : -1,
    'gap penalty' : -2
    }

blosum_scoring_matrix = {
    'type' : 'blosum62',
    'matching matrix' : blosum_62,
    'gap penalty' : -4
    }

# Enumeration class for direction
class Direction():
    LEFT = 1
    UP = 2
    UP_LEFT = 3
    UNDEFINED = 0
    
# Initialise matrices
def initialize_matrices(seq1, seq2, gap_penalty, method:str='needleman_wunsch'):
    n, m = len(seq1) + 1, len(seq2) + 1
    score_matrix = np.zeros((m, n))
    direction_matrix = np.zeros((m, n))
    
    if method == 'needleman_wunsch':
        score_matrix[:,0] = np.arange(0, m*gap_penalty, gap_penalty) # first col
        score_matrix[0,:] = np.arange(0, n*gap_penalty, gap_penalty) # first row
        
        direction_matrix[1:,0] = Direction.UP
        direction_matrix[0,1:] = Direction.LEFT
        
    if method == 'smith_waterman':
        score_matrix = np.full((m,n), Direction.UNDEFINED)
    
    return score_matrix, direction_matrix

def calc_match_score(a:str, b:str, scoring_matrix):
    # if using the basic scoring matrix
    if scoring_matrix['type'] == 'basic':
        match_score = scoring_matrix['match score']
        mismatch_score = scoring_matrix['mismatch score']
        
        return (match_score if a == b else mismatch_score )
    
    # if using BLOSUM62 scoring matrix
    if scoring_matrix['type'] == 'blosum62': 
        blosum_62 = scoring_matrix['matching matrix']
        return blosum_62[a][b]
    
# Global Alignment - Needleman-Wunsch algorithm
def needleman_wunsch(seq1, seq2, scoring_matrix= basic_scoring_matrix):
    """Perform global alignment using the Needleman-Wunsch algorithm."""
    
    n, m = len(seq1) + 1, len(seq2) + 1
    
    gap_penalty = scoring_matrix['gap penalty']
    
    # if scoring_matrix['type'] == 'basic':
    #     match_score = scoring_matrix['match score']
    #     mismatch_score = scoring_matrix['mismatch score']

    # if scoring_matrix['type'] == 'blosum62':
    #     blosum_62 = scoring_matrix['matching matrix']
             
    score_matrix, direction_matrix = initialize_matrices(seq1, seq2, 
                                                         gap_penalty,
                                                         method= 'needleman_wunsch')
    
    for i in range(1, m):
        for j in range(1, n):
            # Cell Scores
            s_left   = score_matrix[i,   j-1]
            s_up     = score_matrix[i-1,   j]
            s_upleft = score_matrix[i-1, j-1]
            
            hs = s_left + gap_penalty
            vs = s_up + gap_penalty
            ms = s_upleft + calc_match_score(seq1[j-1], seq2[i-1], scoring_matrix)
    
            # # if using the basic scoring matrix
            # if scoring_matrix['type'] == 'basic':
            #     ms = s_upleft + (match_score if seq1[j-1] == seq2[i-1] 
            #                      else mismatch_score )
            
            # # if using BLOSUM62 scoring matrix
            # if scoring_matrix['type'] == 'blosum62':
            #     match_score = blosum_62[seq1[j-1]][seq2[i-1]]
            #     ms = s_upleft + match_score
                            
            score_matrix[i,j] = max(hs, vs, ms)
            
            if score_matrix[i,j] == ms:
                direction_matrix[i,j] = Direction.UP_LEFT
            elif score_matrix[i,j] == vs:
                direction_matrix[i,j] = Direction.UP
            else:
                direction_matrix[i,j] = Direction.LEFT
        
    return score_matrix, direction_matrix

def nw_traceback(seq1, seq2, score_matrix, direction_matrix):
    """Perform traceback after using Needleman-Wunsch algorithm."""
    i, j = len(seq2), len(seq1)
    res1 = []
    res2 = []
    
    while i > 0 or j > 0:
        if direction_matrix[i,j] == Direction.UP_LEFT:
            res1.append(seq1[j-1])
            res2.append(seq2[i-1])
            i -= 1
            j -= 1
        elif direction_matrix[i,j] == Direction.UP:
            res1.append('-')        # introduce a gap on top when vertical
            res2.append(seq2[i-1])
            i -= 1
        else:
            res1.append(seq1[j-1])  # introduce a gap on left when horizontal
            res2.append('-')
            j -= 1
    
    # reverse the sequence
    res1 = ''.join(res1)[::-1]
    res2 = ''.join(res2)[::-1]
    return res1, res2
    
#  Local Alignment - Smith-Waterman algorithm
def smith_waterman(seq1, seq2, gap_penalty, scoring_matrix= basic_scoring_matrix):
    """Perform local alignment using the Smith-Waterman algorithm."""
    
    # Extract scoring 
    if scoring_matrix['type'] == 'basic':
        match_score = scoring_matrix['match score']
        mismatch_score = scoring_matrix['mismatch score']

    if scoring_matrix['type'] == 'blosum62':
        blosum_62 = scoring_matrix['matching matrix']
        gap_penalty = scoring_matrix['gap penalty']
        
    score_matrix, direction_matrix = initialize_matrices(seq1, seq2, 
                                                         gap_penalty= 0,
                                                         method= 'smith_waterman')
    
    n, m = len(seq1) + 1, len(seq2) + 1
    max_score, max_pos = 0, (0,0)

    for i in range(1, m):
        for j in range(1, n):
            # Cell Scores
            s_left   = score_matrix[i,   j-1]
            s_up     = score_matrix[i-1,   j]
            s_upleft = score_matrix[i-1, j-1]
            
            hs = s_left + gap_penalty
            vs = s_up + gap_penalty
            
            # if using the basic scoring matrix
            if scoring_matrix['type'] == 'basic':
                ms = s_upleft + (match_score if seq1[j-1] == seq2[i-1] 
                                 else mismatch_score)
            
            # if using BLOSUM62 scoring matrix
            if scoring_matrix['type'] == 'blosum62':
                match_score = blosum_62[seq1[j-1]][seq2[i-1]]
                ms = s_upleft + match_score
                
            score = max(hs, vs, ms)
            
            if score < 0:
                score_matrix[i,j] = 0
            else:
                score_matrix[i,j] = score
                if score_matrix[i,j] == ms:
                    direction_matrix[i,j] = Direction.UP_LEFT
                elif score_matrix[i,j] == vs:
                    direction_matrix[i,j] = Direction.UP
                else:
                    direction_matrix[i,j] = Direction.LEFT
    
    return score_matrix, direction_matrix

def sw_traceback(seq1, seq2, score_matrix, direction_matrix):
    index = np.unravel_index(np.argmax(score_matrix), np.shape(score_matrix))
    i, j = index
    res1 = []
    res2 = []
    
    while score_matrix[i,j] != 0:
        if direction_matrix[i,j] == Direction.UP_LEFT:
            res1.append(seq1[j-1])
            res2.append(seq2[i-1])
            i -= 1
            j -= 1
        elif direction_matrix[i,j] == Direction.UP:
            res1.append('-')        # introduce a gap on top when vertical
            res2.append(seq2[i-1])
            i -= 1
        else:
            res1.append(seq1[j-1])  # introduce a gap on left when horizontal
            res2.append('-')
            j -= 1

    # reverse the sequence
    res1 = ''.join(res1)[::-1]
    res2 = ''.join(res2)[::-1]
    return res1, res2

def alignment_score(res1, res2, method:str):
    if method == 'needleman_wunsch':
        misalignment_score = -2
    elif method == 'smith_waterman':
        misalignment_score = -1
    
    alignment_score = sum(1 for a,b in zip(res1, res2) if a == b)
    percentage_identity = alignment_score / len(res1) * 100
    alignment_score += sum(misalignment_score for a,b in zip(res1, res2) if a != b)
    
    return alignment_score, percentage_identity
        
def align(seq1, seq2, method:str, scoring_type):
    
    if scoring_type == 'basic':
        scoring_matrix = basic_scoring_matrix
        
    if scoring_type == 'blosum62':
        scoring_matrix = blosum_scoring_matrix
    
    if method == 'needleman_wunsch':
        score_matrix, direction_matrix = needleman_wunsch(seq1, seq2, scoring_matrix)
        res1, res2 = nw_traceback(seq1, seq2, score_matrix, direction_matrix)
        align_score, percent_iden = alignment_score(res1, res2, method)
        
    if method == 'smith_waterman':
        score_matrix, direction_matrix = smith_waterman(seq1, seq2, 
                                                        gap_penalty= -1,
                                                        scoring_matrix= scoring_matrix)
        res1, res2 = sw_traceback(seq1, seq2, score_matrix, direction_matrix)
        align_score, percent_iden = alignment_score(res1, res2, method= 'smith_waterman')

    name = method.replace('-', ' ').title()
    print('====================')
    print('Method: ' + name)
    print('Scoring Type: ' + scoring_type)
    print('Input Strings')
    print(seq1)
    print(seq2)
    print()
    print('Result')
    print(res1)
    print(res2)
    print()
    print('Alignment Score:', align_score)
    print('Percentage Identity:', percent_iden, '%')
    print('====================')
        

# Input sequences
seq1 = 'MANFLLPRGTSSFRRFTRES'   # input string of amino acids sequence
seq2 = 'MWWFLLPRGTRRFTMLS'      # input string of amino acids sequence

# seq1 = 'MEELDAQY'   
# seq2 = 'MENSELAQY'

# seq1 = 'GMFPQ'   
# seq2 = 'PWWGMTFPMH'

# First Tasks - Global & Local Alignment
# Percentage Identity & Overall Score
# align('MEELDAQY', 'MENSELAQY', 'needleman_wunsch', 'basic')

# align('GMFPQ', 'PWWGMTFPMH', 'smith_waterman', 'basic')

align('MANFLLPRGTSSFRRFTRES', 'MWWFLLPRGTRRFTMLS', 'needleman_wunsch', 'basic')

# Second Task - Using BLOSUM62 Matrix
align('MANFLLPRGTSSFRRFTRES', 'MWWFLLPRGTRRFTMLS', 'needleman_wunsch', 'blosum62')
align('MANFLLPRGTSSFRRFTRES', 'MWWFLLPRGTRRFTMLS', 'smith_waterman', 'blosum62')

# Final Tasks - Align and Compare 2 Isoforms
isoform_i = 'MANFLLPRGTSSFRRFTRESLAAIEKRMAEKQARGSTTLQESREGLPEEEAPRPQL\
DLQASKKLPDLYGNPPQELIGEPLEDLDPFYSTQKTFIVLNKGKTIFRFSATNALYVLS\
PFHPIRRAAVKILVHSYPLQLIPAEYPLGAHAGDVHVLPPPHLHSSLGVRVALIPCWS'
        
isoform_j = 'MANFLLPRGTSSFRRFTRESLAAIEKRMAEKQARGSTTLQESREGLPEEEAPRPQL\
DLQASKKLPDLYGNPPQELIGEPLEDLDPFYSTQKVTTTHLQPCLPFCATPLSMEQRGK\
RAWPPYGALFRVAHEALGK'
            
align(isoform_i, isoform_j, 'needleman_wunsch', 'blosum62')
align(isoform_i, isoform_j, 'smith_waterman', 'blosum62')

# score_matrix, direction_matrix = needleman_wunsch(seq1, seq2)
# res1, res2 = nw_traceback(seq1, seq2, score_matrix, direction_matrix)
# align_score, percent_iden = alignment_score(res1, res2, 'needleman_wunsch')




