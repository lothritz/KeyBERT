#Original Code by https://github.com/esantus/APSyn

def APSyn(x_row, y_row):
    """
    APSyn(x, y) = (\sum_{f\epsilon N(f_{x})\bigcap N(f_{y})))} \frac{1}{(rank(f_{x})+rank(f_{y})/2)})
    :param x_row:
    :param y_row:
    :return:
    """

    # Sort y's contexts
    y_contexts_cols = sort_by_value_get_col(y_row) # tuples of (row, col, value)
    y_context_rank = { c : i + 1 for i, c in enumerate(y_contexts_cols) }

    # Sort x's contexts
    x_contexts_cols = sort_by_value_get_col(x_row)
    assert len(x_contexts_cols) == len(y_contexts_cols)

    x_context_rank = { c : i + 1 for i, c in enumerate(x_contexts_cols) }

    # Average of 1/(rank(w1)+rank(w2)/2) for every intersected feature among the top N contexts
    intersected_context = set(y_contexts_cols).intersection(set(x_contexts_cols))

    # if formula == F_ORIGINAL:
    score_original = sum([2.0 / (x_context_rank[c] + y_context_rank[c]) for c in intersected_context]) #Original

    #if formula == F_POWER:
    #    score_power = sum([2.0 / (math.pow(x_context_rank[c], POWER) + math.pow(y_context_rank[c], POWER)) for c in intersected_context])
    #elif formula == F_BASE_POWER:
    #    score_power = sum([math.pow(BASE, (x_context_rank[c]+y_context_rank[c])/2.0) for c in intersected_context])
    #else:
    score_power = score_original
        # sys.exit('Formula value not found!')

    return score_original, score_power


def sort_by_value_get_col(mat):
    """
    Sort a sparse coo_matrix by values and returns the columns (the matrix has 1 row)
    :param mat: the matrix
    :return: a sorted list of tuples columns by descending values
    """
    sorted_tuples = sorted(mat, key=lambda x: x[2], reverse=True)

    if len(sorted_tuples) == 0:
        return []

    rows, columns, values = zip(*sorted_tuples)
    return columns