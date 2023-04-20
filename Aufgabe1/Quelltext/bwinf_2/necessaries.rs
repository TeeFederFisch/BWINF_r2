use std::fs::File;
use std::io::Write;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Dot {
    pub x: f32,
    pub y: f32
}

impl Dot {
    // for debugging
    pub fn _to_string(&self) -> String {
        format!("Dot {{ x: {}, y: {}}}", self.x, self.y)
    }
    // for converting back to (f32, f32)
    pub fn get_infos(&self) -> (f32, f32) {
        (self.x, self.y)
    }

    pub fn distance(&self, other: &Dot) -> f32 {
        return ((self.x.max(other.x) - self.x.min(other.x)).powi(2)
                + (self.y.max(other.y) - self.y.min(other.y)).powi(2)).sqrt();
    }
    // initializes the fitting algorithm depending on the mode
    pub fn start<'a>(&'a self, points: &'a Vec<Dot>, mode: Mode) -> Result<(f32, Vec<Dot>), i32> {
        let mut best: (f32, Vec<Dot>) = (f32::MAX, Vec::new());
        let visited: Vec<Dot> = vec![*self];

        let mut buff: Vec<(Vec<Dot>, (Dot, Dot))> = Vec::new();

        for point in points {
            if point == self {continue}
            if let Ok(res) = match mode {
                Mode::Normal => point.check(&points, visited.clone(), *self),
                Mode::Expensive => point.check_all(&points, visited.clone(), *self),
                Mode::Cached => point.check_cached(&points, visited.clone(), *self, &mut buff),
                Mode::Optimal => point.check_optimal(&points, visited.clone(), *self)
            } {
                if res.0 + self.distance(point) < best.0 {
                    best = (res.0 + self.distance(point), res.1);
                    if mode == Mode::Optimal {break;}
                }
            };
        }
        Ok(best)
    }
    
    // responsible for most of the runtime
    fn check<'a>
    (
        &'a self,
        points: &'a Vec<Dot>,
        mut visited: Vec<Dot>,
        last: Dot
    ) -> Result<(f32, Vec<Dot>), i32> {

        visited.push(*self);
        
        // a result is found if theres no points left to visit
        if visited.len() == points.len() {return Ok((0., visited))}

        let mut possible = Vec::new();
        // calculating all the possible next points
        for i in points {
            if !visited.contains(&i) && is_valid(last, *self, *i){
                possible.push((i.distance(self), i));
            }
        }
        // not every point was visited but theres also none to reach
        if possible.is_empty() {return Err(1)}

        possible.sort_by(|a, b| a.0.total_cmp(&b.0));

        let perfect = possible.get(0).unwrap().1;
        
        if let Ok(res) = perfect.check(points, visited, *self) {
            return Ok((res.0 + self.distance(perfect), res.1));
        } else {
            return Err(1)
        }
    }


    fn check_all<'a>
    (
        &'a self,
        points: &'a Vec<Dot>,
        mut visited: Vec<Dot>,
        last: Dot
    ) -> Result<(f32, Vec<Dot>), i32> {

        visited.push(*self);
        
        // a result is found if theres no points left to visit
        if visited.len() == points.len() {return Ok((0., visited))}

        let mut perfect = &Dot { x: 0., y: 0. };
        let mut perfect_path = (f32::MAX, Vec::new());

        for point in points {
            if !visited.contains(&point) && is_valid(last, *self, *point) {
                // "if let" unstable in connection with other expressions, nested if's necessary
                if let Ok(res) = point.check_all(points, visited.clone(), *self) {
                    if res.0 < perfect_path.0 {
                        perfect_path = res;
                        perfect = point;
                    }
                }
            }
        }

        if perfect_path.1.is_empty() {
            return Err(1)
        }

        return Ok((perfect_path.0 + self.distance(perfect), perfect_path.1));
    }

    fn check_optimal<'a>
    (
        &'a self,
        points: &'a Vec<Dot>,
        mut visited: Vec<Dot>,
        last: Dot
    ) -> Result<(f32, Vec<Dot>), i32> {
        
        visited.push(*self);
        
        // a result is found if theres no points left to visit
        if visited.len() == points.len() {return Ok((0., visited))}

        let mut possible = Vec::new();

        for i in points {
            if !visited.contains(&i) && is_valid(last, *self, *i){
                possible.push((i.distance(self), i));
            }
        }

        if possible.is_empty() {return Err(1)}

        possible.sort_by(|a, b| a.0.total_cmp(&b.0));

        for i in 0..possible.len() {
            if possible.len() > i {
                if let Ok(a) = possible[i].1.check_optimal(points, visited.clone(), *self) {
                    return Ok((a.0 + self.distance(possible[i].1), a.1));
                }
            }
        }

        return Err(1)
    }

    fn check_cached<'a>(
        &'a self,
        points: &'a Vec<Dot>,
        mut visited: Vec<Dot>,
        last: Dot,
        buff: &mut Vec<(Vec<Dot>, (Dot, Dot))>
    ) -> Result<(f32, Vec<Dot>), i32> {

        visited.push(*self);
        
        let mut a = visited.clone();
        a.sort_by(|a, b| (a.x, a.y).partial_cmp(&(b.x, b.y)).unwrap());
        if buff.contains(&(a.clone(), (*self, last))) {
            return Err(1)
        }

        // a result is found if theres no points left to visit
        if visited.len() == points.len() {return Ok((0., visited))}
        
        let mut perfect = &Dot { x: 0., y: 0. };
        let mut perfect_path = (f32::MAX, Vec::new());

        for point in points {
            if !visited.contains(&point) && is_valid(last, *self, *point) {
                // "if let" unstable in connection with other expressions, nested if's necessary
                if let Ok(res) = point.check_cached(points, visited.clone(), *self, buff) {
                    if res.0 < perfect_path.0 {
                        perfect_path = res;
                        perfect = point;
                    }
                }
            }
        }

        if perfect_path.1.is_empty() {
            if visited.len() < 5 {
                buff.push((a, (*self, last)));
            }
            return Err(1)
        }

        return Ok((perfect_path.0 + self.distance(perfect), perfect_path.1));
    }

}

pub fn start_cached(points: &Vec<Dot>) -> (f32, Vec<Dot>) {
    let mut buff: Vec<(Vec<Dot>, (Dot, Dot))> = Vec::new();
    let mut results = Vec::new();

    for last in points.clone() {
        let visited: Vec<Dot> = vec![last];
        let mut local_results = Vec::new();
        let mut best: (f32, Vec<Dot>) = (f32::MAX, Vec::new());

        for point in points.clone() {
            if point == last {break}
            if let Ok(res) = point.check_cached(&points, visited.clone(), last, &mut buff) {
                local_results.push((res.clone().0 + last.distance(&point), res.1));
            }
        }
        for res in local_results {
            if res.0 < best.0 {
                best = res;
            }
        }
        results.push(best);
    }

    let mut best: (f32, Vec<Dot>) = (f32::MAX, Vec::new());

    for res in results {
        if res.0 < best.0 {
            best = res;
        }
    }
    best
}

pub fn is_valid(from: Dot, current: Dot, to_check: Dot) -> bool {
    // multiple checks to be safe from dividing through zero
    if current.x == from.x {
        if to_check.y == current.y {return true;}
        return (from.y - current.y).is_sign_positive() == (to_check.y - current.y).is_sign_negative();
    }

    if current.y == from.y {
        if to_check.x == current.x {return true;}
        return (from.x - current.x).is_sign_positive() == (to_check.x - current.x).is_sign_negative();
    }

    let gradient_normal = -1. / ((current.y - from.y) / (current.x - from.x));
    // term of the borderline
    let calc = |x: f32| gradient_normal * (x - current.x) + current.y;

    if calc(from.x) > from.y {
        return calc(to_check.x) <= to_check.y;
    } else {
        return calc(to_check.x) >= to_check.y;
    }
}

// for saving results in files
pub fn save_path(name: String, path: Vec<(f32, f32)>) {
    let mut buf = String::new();
    for point in path {
        buf += format!("{} {}\n", point.0, point.1).as_str();
    }

    File::create(name).unwrap().write(buf.as_bytes()).unwrap();
}

pub enum In {
    AFile(String),
    AVec(Vec<(f32, f32)>)
}

#[derive(Clone, Copy, PartialEq)]
pub enum Mode {
    Expensive,  // -> documentation algorithm nr. 1
    Cached,     // -> documentation algorithm nr. 2
    Normal,     // -> documentation algorithm nr. 3
    Optimal,    // -> documentation algorithm nr. 4
}
impl Mode {
    pub fn to_string(m: Mode) -> String {
        match m {
            Mode::Expensive => "Expensive".to_string(),
            Mode::Normal => "Normal".to_string(),
            Mode::Cached => "Cached".to_string(),
            Mode::Optimal => "Optimal".to_string()
        }
    }
}