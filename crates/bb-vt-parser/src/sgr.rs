//! SGR (Select Graphic Rendition) application onto a pen.

use bb_terminal_model::{CellAttrs, Color};

/// Apply an SGR parameter list to `pen`.
///
/// Each entry in `params` is one semicolon-separated parameter, itself a
/// list of colon-separated sub-parameters (a plain parameter has exactly
/// one entry). An empty list means `SGR 0` (full reset).
pub(crate) fn apply(pen: &mut CellAttrs, params: &[Vec<u16>]) {
    if params.is_empty() {
        *pen = CellAttrs::default();
        return;
    }
    let mut i = 0;
    while i < params.len() {
        let parts = params.get(i).map(Vec::as_slice).unwrap_or(&[]);
        let code = parts.first().copied().unwrap_or(0);
        match code {
            0 => *pen = CellAttrs::default(),
            1 => pen.bold = true,
            2 => pen.dim = true,
            3 => pen.italic = true,
            4 => pen.underline = true,
            5 | 6 => pen.blink = true,
            7 => pen.reverse = true,
            8 => pen.hidden = true,
            9 => pen.strikethrough = true,
            // KNOWN DIVERGENCE: ECMA-48 and xterm ctlseqs both define SGR
            // 21 as *doubly underlined*. We treat it as bold-off instead
            // — the rival convention several terminals (the Linux console
            // among them) implement.
            //
            // The constraint is about underline, not intensity:
            // `CellAttrs::underline` is a plain bool with no underline
            // *style*, so "doubly" has nowhere faithful to go, and adding
            // a field was out of scope here. Intensity itself is fully
            // representable — `bold` and `dim` are independent fields —
            // which is why 22 correctly clears both.
            //
            // Revisit if `CellAttrs` ever grows an underline style.
            21 | 22 => {
                pen.bold = false;
                pen.dim = false;
            }
            23 => pen.italic = false,
            24 => pen.underline = false,
            25 => pen.blink = false,
            27 => pen.reverse = false,
            28 => pen.hidden = false,
            29 => pen.strikethrough = false,
            30..=37 => pen.fg = indexed(code - 30),
            39 => pen.fg = Color::Default,
            40..=47 => pen.bg = indexed(code - 40),
            49 => pen.bg = Color::Default,
            90..=97 => pen.fg = indexed(code - 90 + 8),
            100..=107 => pen.bg = indexed(code - 100 + 8),
            38 | 48 => {
                let color = if parts.len() > 1 {
                    // Colon form: 38:5:n / 38:2[:cs]:r:g:b — all args in
                    // the sub-parameter list of this single parameter.
                    extended_color(parts.get(1..).unwrap_or(&[]))
                } else {
                    // Semicolon form: 38;5;n / 38;2;r;g;b — consumes the
                    // following parameters.
                    let (color, used) = extended_color_semicolon(params, i + 1);
                    i += used;
                    color
                };
                if let Some(color) = color {
                    if code == 38 {
                        pen.fg = color;
                    } else {
                        pen.bg = color;
                    }
                }
            }
            _ => {} // unknown SGR: ignored
        }
        i += 1;
    }
}

fn indexed(n: u16) -> Color {
    Color::Indexed(n.min(255) as u8)
}

fn channel(v: Option<&u16>) -> u8 {
    v.copied().unwrap_or(0).min(255) as u8
}

/// Parse the argument list of a colon-form extended color: `[5, n]`,
/// `[2, r, g, b]`, or the ITU T.416 form `[2, colorspace, r, g, b]`.
fn extended_color(args: &[u16]) -> Option<Color> {
    match args.first()? {
        5 => Some(Color::Indexed(channel(args.get(1)))),
        2 => {
            // With 4+ trailing args the first is a colorspace id.
            let rgb_start = if args.len() >= 5 { 2 } else { 1 };
            let r = channel(args.get(rgb_start));
            let g = channel(args.get(rgb_start + 1));
            let b = channel(args.get(rgb_start + 2));
            Some(Color::Rgb(r, g, b))
        }
        _ => None,
    }
}

/// Parse a semicolon-form extended color starting at `params[at]`
/// (the mode selector). Returns the color (if the arguments are
/// complete) and the number of extra parameters consumed.
fn extended_color_semicolon(params: &[Vec<u16>], at: usize) -> (Option<Color>, usize) {
    let first = |i: usize| params.get(i).and_then(|p| p.first().copied());
    match first(at) {
        Some(5) => {
            let Some(n) = first(at + 1) else {
                return (None, 1);
            };
            (Some(Color::Indexed(n.min(255) as u8)), 2)
        }
        Some(2) => {
            let (Some(r), Some(g), Some(b)) = (first(at + 1), first(at + 2), first(at + 3)) else {
                // Truncated: consume what exists, apply nothing.
                let avail = params.len().saturating_sub(at);
                return (None, avail.min(4));
            };
            (
                Some(Color::Rgb(
                    r.min(255) as u8,
                    g.min(255) as u8,
                    b.min(255) as u8,
                )),
                4,
            )
        }
        Some(_) => (None, 1),
        None => (None, 0),
    }
}
