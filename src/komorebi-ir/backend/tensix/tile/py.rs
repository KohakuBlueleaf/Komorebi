//! The few things of Python's own that today's tile compiler prints or computes with,
//! as CPython 3.12 and later do them: a float's and a text's `repr`, `round(x, 1)`,
//! `sum` of floats (compensated), `float.hex` and the `g` format. The port's texts
//! depend on them byte for byte.

/// `repr(x)` of a float: the shortest digits that read back as `x`; fixed notation
/// when the decimal point falls from 4 places left of the first digit to 16 right of
/// it, else `d.ddde±XX`; a whole number keeps `.0`.
pub fn float_repr(x: f64) -> String {
    if x.is_nan() {
        return "nan".to_string();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf" } else { "-inf" }.to_string();
    }
    let sign = if x.is_sign_negative() { "-" } else { "" };
    if x == 0.0 {
        return format!("{sign}0.0");
    }
    // Rust's `{:e}` gives the same shortest round-trip digits: `d.ddde<exp>`.
    let sci = format!("{:e}", x.abs());
    let (mantissa, exp) = sci.split_once('e').unwrap_or((sci.as_str(), "0"));
    let exp: i32 = exp.parse().unwrap_or(0);
    let digits: String = mantissa.chars().filter(char::is_ascii_digit).collect();
    let point = exp + 1; // the decimal point's place after the first digit
    let n = digits.len() as i32;
    let body = if point <= -4 || point > 16 {
        let (first, rest) = digits.split_at(1);
        let rest = if rest.is_empty() {
            String::new()
        } else {
            format!(".{rest}")
        };
        let e_sign = if exp < 0 { '-' } else { '+' };
        format!("{first}{rest}e{e_sign}{:02}", exp.abs())
    } else if point <= 0 {
        format!("0.{}{digits}", "0".repeat((-point) as usize))
    } else if point >= n {
        format!("{digits}{}.0", "0".repeat((point - n) as usize))
    } else {
        let (whole, part) = digits.split_at(point as usize);
        format!("{whole}.{part}")
    };
    format!("{sign}{body}")
}

/// `repr(s)` of a text: in single quotes, or double ones when it holds a single quote
/// and no double one; backslashes, the quote, tabs, newlines, returns and other
/// unprintable characters escaped.
pub fn str_repr(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x20 || c as u32 == 0x7f => {
                out.push_str(&format!("\\x{:02x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

/// `round(x, 1)`: the nearest tenth of the exact value, ties to even (Rust's `{:.1}`
/// rounds the exact value the same way), read back as a float.
pub fn round1(x: f64) -> f64 {
    if !x.is_finite() {
        return x;
    }
    format!("{x:.1}").parse().unwrap_or(x)
}

/// `sum(xs)` of floats as CPython 3.12 and later compute it: Neumaier's compensated
/// sum from 0.0, the compensation added at the end when it is not zero and finite.
pub fn fsum(xs: impl IntoIterator<Item = f64>) -> f64 {
    let (mut total, mut c) = (0.0f64, 0.0f64);
    for x in xs {
        let t = total + x;
        if total.abs() >= x.abs() {
            c += (total - t) + x;
        } else {
            c += (x - t) + total;
        }
        total = t;
    }
    if c != 0.0 && c.is_finite() {
        total += c;
    }
    total
}

/// `x.hex()`: `[-]0x1.<13 hex digits>p<exponent>` (`0x0.` for a subnormal, and
/// `0x0.0p+0` for zero); `inf`, `-inf` and `nan` for the others.
pub fn float_hex(x: f64) -> String {
    if x.is_nan() {
        return "nan".to_string();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf" } else { "-inf" }.to_string();
    }
    let sign = if x.is_sign_negative() { "-" } else { "" };
    if x == 0.0 {
        return format!("{sign}0x0.0p+0");
    }
    let bits = x.to_bits();
    let mantissa = bits & ((1u64 << 52) - 1);
    let biased = ((bits >> 52) & 0x7ff) as i64;
    let (lead, exponent) = if biased == 0 {
        (0, -1022)
    } else {
        (1, biased - 1023)
    };
    let e_sign = if exponent < 0 { '-' } else { '+' };
    format!("{sign}0x{lead}.{mantissa:013x}p{e_sign}{}", exponent.abs())
}

/// `format(x, ".<precision>g")`: `precision` significant digits, fixed notation when
/// the rounded value's exponent is from -4 to below `precision`, else `d.ddde±XX`;
/// trailing zeros and a bare point dropped.
pub fn format_g(x: f64, precision: usize) -> String {
    if x.is_nan() {
        return "nan".to_string();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf" } else { "-inf" }.to_string();
    }
    let p = precision.max(1);
    if x == 0.0 {
        return if x.is_sign_negative() { "-0" } else { "0" }.to_string();
    }
    let sci = format!("{:.*e}", p - 1, x);
    let (mantissa, exp) = sci.split_once('e').unwrap_or((sci.as_str(), "0"));
    let exp: i64 = exp.parse().unwrap_or(0);
    let trim = |t: &str| -> String {
        if t.contains('.') {
            t.trim_end_matches('0').trim_end_matches('.').to_string()
        } else {
            t.to_string()
        }
    };
    if -4 <= exp && exp < p as i64 {
        let decimals = (p as i64 - 1 - exp).max(0) as usize;
        trim(&format!("{x:.decimals$}"))
    } else {
        let e_sign = if exp < 0 { '-' } else { '+' };
        format!("{}e{e_sign}{:02}", trim(mantissa), exp.abs())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floats_print_as_python_prints_them() {
        let cases = [
            (1.0, "1.0"),
            (120.3, "120.3"),
            (2.2825186, "2.2825186"),
            (1e16, "1e+16"),
            (1e15, "1000000000000000.0"),
            (0.0001, "0.0001"),
            (0.00001, "1e-05"),
            (-1.42860677e-06, "-1.42860677e-06"),
            (1.5e17, "1.5e+17"),
            (-0.0, "-0.0"),
            (0.1 + 0.2, "0.30000000000000004"),
            (8388608.0, "8388608.0"),
        ];
        for (x, want) in cases {
            assert_eq!(float_repr(x), want, "{x}");
        }
    }

    #[test]
    fn texts_and_rounding_and_sums_are_pythons() {
        assert_eq!(str_repr("0.01"), "'0.01'");
        assert_eq!(str_repr("it's"), "\"it's\"");
        assert_eq!(str_repr("a'\"b"), "'a\\'\"b'");
        assert_eq!(
            [0.25, 0.75, 1.25, 0.05, 0.15, 120.25].map(round1),
            [0.2, 0.8, 1.2, 0.1, 0.1, 120.2]
        );
        // sum([0.1] * 10) is 1.0 in Python 3.12 (compensated), 0.9999999999999999
        // added one by one
        assert_eq!(fsum([0.1; 10]), 1.0);
        assert_eq!(fsum([1e100, 1.0, -1e100, 1.0]), 2.0);
    }

    #[test]
    fn hex_and_g_are_pythons() {
        let hex = [
            (2.0, "0x1.0000000000000p+1"),
            (0.5, "0x1.0000000000000p-1"),
            (-8388608.0, "-0x1.0000000000000p+23"),
            (f64::from(std::f32::consts::LN_2), "0x1.62e4300000000p-1"),
            (1e-40f32 as f64, "0x1.16c2000000000p-133"),
            (0.01f32 as f64, "0x1.47ae140000000p-7"),
            (-0.0, "-0x0.0p+0"),
            (0.0, "0x0.0p+0"),
            (f64::from_bits(1), "0x0.0000000000001p-1022"),
        ];
        for (x, want) in hex {
            assert_eq!(float_hex(x), want, "{x}");
        }
        let g = [
            (3e-7, 3, "3e-07"),
            (1.75e-3, 3, "0.00175"),
            (f64::INFINITY, 6, "inf"),
            (f64::NEG_INFINITY, 6, "-inf"),
            (-87.0, 6, "-87"),
            (0.0, 6, "0"),
            (-0.0, 6, "-0"),
            (123456789.0, 6, "1.23457e+08"),
            (0.0001, 6, "0.0001"),
            (0.00001234, 6, "1.234e-05"),
            (999999.5, 6, "1e+06"),
            (9.9995, 4, "9.999"),
        ];
        for (x, p, want) in g {
            assert_eq!(format_g(x, p), want, "{x} {p}");
        }
    }
}
