//! OSC (Operating System Command) payload parsing.

use crate::VtEvent;

/// Parse a complete OSC payload (everything between `ESC ]` and the
/// terminator) and return the resulting semantic event, if any.
///
/// Recognized codes: 0/2 (title), 7 (working directory), 8 (hyperlink),
/// 133 (prompt protocol). Everything else is ignored. The payload is
/// decoded as UTF-8 with invalid bytes replaced by U+FFFD.
///
/// A payload yields at most one event; the caller stamps it with the
/// cursor position at the terminator byte.
pub(crate) fn dispatch(buf: &[u8]) -> Option<VtEvent> {
    let s = String::from_utf8_lossy(buf);
    let (code, rest) = match s.split_once(';') {
        Some((code, rest)) => (code, Some(rest)),
        None => (s.as_ref(), None),
    };
    let code = code.parse::<u32>().ok()?;
    match code {
        0 | 2 => Some(VtEvent::Title(rest.unwrap_or("").to_owned())),
        7 => Some(VtEvent::WorkingDirectory(rest.unwrap_or("").to_owned())),
        8 => hyperlink(rest),
        133 => prompt_marker(rest),
        _ => None, // unknown OSC: ignored
    }
}

/// OSC 8 — `8;params;uri`. An empty URI closes the hyperlink.
fn hyperlink(rest: Option<&str>) -> Option<VtEvent> {
    // `None` rest is "8" with no payload, and a missing params/uri
    // separator is equally malformed: both ignored.
    let (_params, uri) = rest?.split_once(';')?;
    let link = if uri.is_empty() {
        None
    } else {
        Some(uri.to_owned())
    };
    Some(VtEvent::Hyperlink(link))
}

/// OSC 133 — the shell-integration prompt protocol (`A`/`B`/`C`/`D`).
fn prompt_marker(rest: Option<&str>) -> Option<VtEvent> {
    let rest = rest?; // bare "133": ignored
    let (cmd, arg) = match rest.split_once(';') {
        Some((cmd, arg)) => (cmd, Some(arg)),
        None => (rest, None),
    };
    match cmd {
        "A" => Some(VtEvent::PromptStart),
        "B" => Some(VtEvent::CommandStart),
        "C" => Some(VtEvent::CommandExecuted),
        "D" => {
            // The exit code may be followed by extra `;key=value` fields
            // (as some shells send); only the first segment is the code.
            let exit_code = arg
                .and_then(|a| a.split(';').next())
                .and_then(|c| c.parse::<i32>().ok());
            Some(VtEvent::CommandFinished { exit_code })
        }
        _ => None, // unknown 133 subcommand: ignored
    }
}
