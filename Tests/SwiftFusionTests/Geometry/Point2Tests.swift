//
//  Point2Tests.swift
//  
//
//  Created by Fan Jiang on 2020/3/29.
//

@testable import SwiftFusion
import XCTest

final class Point2Tests: XCTestCase {

  func testBetweenIdentities() {
    let a = Point2(0,0), b = Point2(1,1)
    let ab = [a, b]
    let f1: @differentiable (_ ab: [Point2]) -> Point2 = { ab in
      ab[1] - ab[0]
    }
    let H = jacobian(of: f1, at: ab)
    for i in 0..<2 {
      print(H[i][0].recursivelyAllKeyPaths(to:Double.self).map { H[i][1][keyPath: $0] })
    }
  }
  
  static var allTests = [
    ("testBetweenIdentities", testBetweenIdentities)
  ]
}
