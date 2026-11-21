// Signed integer update lanes. All weights and accumulators are runtime inputs.
// One registered output update per cycle given independent accumulator inputs.
// Accumulators wrap mod 2^32, like a 32-bit integer dot-product partial sum.

module mac4x8(input clk, input signed [3:0] w, input signed [7:0] a,
              input signed [31:0] acc, output reg signed [31:0] y);
    wire signed [11:0] product = w * a;
    always @(posedge clk) y <= acc + product;
endmodule

module packed_mac4x8(input clk, input signed [3:0] w0, w1, input signed [7:0] a,
                      input signed [31:0] acc0, acc1, output reg signed [31:0] y0, y1);
    // Two W4A8 products sharing one activation. Use 12 bits per signed product.
    wire signed [16:0] lo = w0, hi = w1;
    wire signed [16:0] packed_weights = lo + (hi <<< 12);
    wire signed [24:0] product = packed_weights * a;
    wire signed [11:0] p0 = product[11:0];
    wire signed [11:0] p1 = $signed(product[23:12]) + $signed({11'b0, product[11]});
    always @(posedge clk) begin
        y0 <= acc0 + p0;
        y1 <= acc1 + p1;
    end
endmodule

module packed_mac4x4(input clk, input signed [3:0] w0, w1, w2, a,
                      input signed [31:0] acc0, acc1, acc2,
                      output reg signed [31:0] y0, y1, y2);
    // One exact signed multiply with three weights sharing one activation.
    // Each product fits eight bits; sum the sign-extended shifted weights.
    wire signed [20:0] lo = w0, mid = w1, hi = w2;
    wire signed [20:0] packed_weights = lo + (mid <<< 8) + (hi <<< 16);
    wire signed [24:0] product = packed_weights * a;
    wire signed [7:0] p0 = product[7:0];
    // A negative lower digit borrows one from the next digit. For p2,
    // test the raw middle digit's sign, which includes the p0 borrow.
    wire signed [7:0] p1 = $signed(product[15:8]) + $signed({7'b0, product[7]});
    wire signed [7:0] p2 = $signed(product[23:16]) + $signed({7'b0, product[15]});
    always @(posedge clk) begin
        y0 <= acc0 + p0;
        y1 <= acc1 + p1;
        y2 <= acc2 + p2;
    end
endmodule

module binary_event(input clk, input signed [3:0] w,
                     input signed [31:0] acc, output reg signed [31:0] y);
    // Issued for an active row.
    always @(posedge clk) y <= acc + w;
endmodule

module level_event(input clk, input signed [3:0] w, level,
                    input signed [31:0] acc, output reg signed [31:0] y);
    wire signed [6:0] weight = w;
    reg signed [6:0] magnitude;
    always @* begin
        case (level)
             4'sd1, -4'sd1: magnitude = weight;
             4'sd2, -4'sd2: magnitude = weight <<< 1;
             4'sd3, -4'sd3: magnitude = (weight <<< 1) + weight;
             4'sd4, -4'sd4: magnitude = weight <<< 2;
             default: magnitude = 0;
        endcase
    end
    wire signed [6:0] delta = level[3] ? -magnitude : magnitude;
    always @(posedge clk) y <= acc + delta;
endmodule

module zero_skip_mac4x4(input clk, input signed [3:0] w, a,
                        input signed [31:0] acc, output reg signed [31:0] y,
                        output weight_read);
    wire signed [7:0] product = w * a;
    assign weight_read = a != 0;
    always @(posedge clk) y <= weight_read ? acc + product : acc;
endmodule
