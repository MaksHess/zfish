import io
from functools import partial
from itertools import islice, pairwise, zip_longest
from typing import TYPE_CHECKING, Any, Callable, Sequence

from zfish.roi._spatial_roi_config import SortKey

if TYPE_CHECKING:
    from zfish.roi.spatial_roi import Roi

DIMS = ("roi", "l", "c", "z", "y", "x")
CAT_DIMS = ("roi", "l", "c")
SPATIAL_DIMS = ("z", "y", "x")
SORT_KEY = SortKey()

def multi_column_print(
    *cols: Sequence[str],
    sep=" ",
    col_width: int | None = None,
    left_padding="",
    right_padding="",
    is_bracket_padding=False,
    indent=0,
    first_indent=True,
):

    n_cols = len(cols)
    n_rows = len(cols[0])
    indent_string = " " * indent
    left_padding_space = " " * len(left_padding)
    right_padding_space = " " * len(right_padding)

    if col_width is None:  # Dynamic column width
        widths = []
        for col in cols:
            try:
                widths.append(max(map(len, col)))
            except ValueError:
                widths = [1] * n_cols
                break
    else:
        widths = [col_width] * n_cols
        cols = map(partial(replace_too_long_elements, col_width=col_width), cols)

    buffer = io.StringIO()

    for i, row in enumerate(zip_longest(*cols, fillvalue="")):
        if first_indent or i != 0:
            print(indent_string, end="", file=buffer)
        if not is_bracket_padding or i == 0:
            print(left_padding, end="", file=buffer)
        else:
            print(left_padding_space, end="", file=buffer)
        for j, (col_row, width) in enumerate(zip(row, widths)):
            if j == n_cols - 1:
                if not is_bracket_padding or i == (n_rows - 1):
                    print(str.ljust(col_row, width), end=right_padding, file=buffer)
                else:
                    print(
                        str.ljust(col_row, width), end=right_padding_space, file=buffer
                    )
            else:
                print(str.ljust(col_row, width), end=sep, file=buffer)
        print(file=buffer)
    return buffer.getvalue()


def replace_too_long_elements(seq: Sequence[str], col_width: int | None = None):
    if col_width is None:
        return seq
    else:
        return [e if len(e) <= col_width else f"{e[:(col_width-1)]}." for e in seq]

def replace_too_long_element(e: str, col_width: int):
    return e if len(e) <= col_width else f"{e[:(col_width-1)]}."

def sequence_to_columns(seq: Sequence[Any], n: int = 3):
    return [list(islice(seq, i, None, n)) for i in range(n)]


def sequence_to_rows(seq: Sequence[Any], n: int = 3):
    import more_itertools
    return list(more_itertools.chunked(seq, n))


def arrange_sequence_in_rows_or_columns(
    raw_sequence: Sequence[Any],
    n: int = 6,
    width: int | None = None,
    indent: int = 0,
    first_indent: bool = True,
    sep: str = "  ",
    left_padding: str = "",
    right_padding: str = "",
    is_bracket_padding: bool = False,
    in_rows: bool = True,
) -> str:
    if len(raw_sequence) == 0:
        return ""
    if in_rows:
        columns = sequence_to_rows(raw_sequence, n=n)
    else:
        columns = sequence_to_columns(raw_sequence, n=n)
    return multi_column_print(
        *columns,
        col_width=width,
        indent=indent,
        first_indent=first_indent,
        sep=sep,
        left_padding=left_padding,
        right_padding=right_padding,
        is_bracket_padding=is_bracket_padding,
    )


def _get_number_of_cols_setting():
    return dict(l=4, c=4)


def _coordinates_repr(
    roi: "Roi",
    col_width=None,
    n_cols: int | dict[str, int] = _get_number_of_cols_setting(),
) -> str:
    buffer = io.StringIO()

    for dim, coord in roi.coords.items():
        print(_general_coord_repr(dim, coord), end="", file=buffer)
        if dim in SPATIAL_DIMS:
            print(_spatial_coord_repr(dim, coord), end="", file=buffer)
        elif dim == 'l':
            print(_label_coord_repr(dim, coord, tables=roi.tables), end='', file=buffer)
            print(file=buffer)
        else:
            print(_cat_coord_repr(dim, coord, col_width=col_width, n_cols=n_cols), end="", file=buffer)
            print(file=buffer)
    return buffer.getvalue()


def _general_coord_repr(dim, coord) -> str:
    is_degenerate = coord.shape == tuple()

    return "  {star} ({name}) ".format(star=(" " if is_degenerate else "*"), name=dim)


def _spatial_coord_repr(dim, coord) -> str:
    is_degenerate = coord.shape == tuple()
    is_empty = coord.shape == (0,)

    if is_empty:
        return ""
    elif is_degenerate:
        return f"{coord.item()}"
    else:
        return f"[{coord.min().item()}-{coord.max().item()}]"


def _cat_coord_repr(
    dim,
    coord,
    col_width=None,
    n_cols: int | dict[str, int] = _get_number_of_cols_setting(),
    first_indent=False
) -> str:
    is_empty = coord.shape == (0,)
    is_degenerate = coord.shape == tuple()
    if is_degenerate:
        values = repr(coord.item())
    else:
        sort_key = SORT_KEY.images if dim == 'c' else SORT_KEY.labels
        values = list(map(repr, sorted(coord.values, key=sort_key)))

    if is_empty:
        return ""
    elif is_degenerate:
        return f"{coord.item()}"
    else:
        n = n_cols if isinstance(n_cols, int) else n_cols[dim]
        return arrange_sequence_in_rows_or_columns(values, indent=8, sep=' ', right_padding=' ', first_indent=first_indent, n=n, in_rows=False, width=col_width)

def _table_coord_repr(dim, coord, tables, is_degenerate) -> str:
    if is_degenerate:
        value = coord.item()
        table = tables[value]

        return f"{repr(coord.item())} (o: {getattr(table, 'height', 'lazy')}, f: {table.width})"
    else:

        values = sorted(coord.values, key=SORT_KEY.labels)
        tables = [tables[value] for value in values]
        max_o = len(str(max(getattr(table, 'height', 'lazy') for table in tables)))
        max_f = len(str(max(table.width for table in tables)))
        repr_values = [repr(value) for value in values]
        repr_tables = [f"(o: {getattr(table, 'height', 'lazy'):>{max_o}}, f: {table.width:>{max_f}})" for table in tables]
        return multi_column_print(repr_values, repr_tables, indent=8, first_indent=False, sep=' ')

def _label_coord_repr(
        dim,
        coord,
        col_width=None,
        n_cols= _get_number_of_cols_setting(),
        tables=None,
) -> str:
    if tables is None or tables == dict():
        return _cat_coord_repr(dim=dim, coord=coord, col_width=col_width, n_cols=n_cols)
    
    is_degenerate = coord.shape == tuple()
    is_empty = coord.shape == (0,)

    if is_empty:
        return ""
    if is_degenerate:
        if coord.item() in tables.keys():
            return _table_coord_repr(dim=dim, coord=coord, tables=tables, is_degenerate=is_degenerate)
        return _cat_coord_repr(dim=dim, coord=coord, col_width=col_width, n_cols=n_cols)
    else:
        coord_with_tables = coord.sel(l=sorted(set(coord.values).intersection(tables.keys())))
        coord_without_tables = coord.sel(l=sorted(set(coord.values).difference(tables.keys())))
        return ''.join([
            _table_coord_repr(dim=dim, coord=coord_with_tables, tables=tables, is_degenerate=is_degenerate), 
            _cat_coord_repr(dim=dim, coord=coord_without_tables, col_width=col_width, n_cols=n_cols, first_indent=True),
            ])


def conditional_separator_join(
    seq: Sequence[str],
    sep: str = ", ",
    false_sep: str | None = None,
    condition: Callable = lambda a, b: not (str.isspace(a) and str.isspace(b)),
) -> str:
    if false_sep is None:
        false_sep = " " * len(sep)
    res = io.StringIO()
    res.write(next(iter(seq)))
    # for a, b in zip(seq, islice(seq, 1, None, None)):
    for a, b in pairwise(seq):
        if condition(a, b):
            res.write(sep)
        else:
            res.write(false_sep)
        res.write(b)
    return res.getvalue()

# def _coordinates_repr(
#     roi: Roi,
#     col_width=None,
#     n_cols: int | dict[str, int] = _get_number_of_cols_setting(),
# ) -> str:
#     "If `col_width` is `None` dynamically set it so all elements can be printed"
#     buffer = io.StringIO()
#     coords_map = {
#         k: v for k, v in chain(roi.labels.coords.items(), roi.images.coords.items())
#     }
#     for dim in sorted(set(DIMS).intersection(coords_map), key=DIMS.index):
#         coord = coords_map[dim]
#         degenerate_dim = coord.shape == tuple()
#         empty_dim = coord.shape == (0,)
#         if not empty_dim:
#             print(
#                 "  {star} ({name}) ".format(
#                     star=(" " if degenerate_dim else "*"), name=dim
#                 ),
#                 end="",
#                 file=buffer,
#             )
#         if dim in SPATIAL_DIMS:
#             if not degenerate_dim:
#                 print(
#                     f"[{coord.min().item()}-{coord.max().item()}]", end="", file=buffer
#                 )
#             else:
#                 print(f"{coord.item()}", end="", file=buffer)
#         else:
#             if not degenerate_dim:
#                 if col_width is None:
#                     try:
#                         dynamic_col_width = (
#                             max(map(len, coord.values)) + 2
#                         )  # since we're printing the sting repr ('')
#                     except ValueError:
#                         dynamic_col_width = 0
#                 else:
#                     dynamic_col_width = col_width

#                 n = n_cols if isinstance(n_cols, int) else n_cols[dim]
#                 print(
#                     f"{arrange_sequence_in_rows_or_columns(list(map(repr, coord.values)), indent=8, sep=' ', right_padding=' ', first_indent=False, n=n, in_rows=False)}",
#                     file=buffer,
#                 )
#             else:
#                 print(repr(coord.item()), file=buffer)
#             # print(file=buffer)
#     return buffer.getvalue()