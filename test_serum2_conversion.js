#!/usr/bin/env node

/**
 * Test script to convert a Serum 2 preset using the node-serum2-preset-packager
 */

const fs = require("fs");
const path = require("path");

// Try to import the package
try {
  const { unpack } = require("node-serum2-preset-packager");
  console.log("Successfully imported node-serum2-preset-packager");

  // Test function
  async function testConversion() {
    const testPresetPath = process.argv[2];
    const outputJsonPath = process.argv[3] || "test_output.json";

    if (!testPresetPath) {
      console.log(
        "Usage: node test_serum2_conversion.js <preset.SerumPreset> [output.json]"
      );
      process.exit(1);
    }

    if (!fs.existsSync(testPresetPath)) {
      console.error(`Preset file not found: ${testPresetPath}`);
      process.exit(1);
    }

    try {
      console.log(`Converting ${testPresetPath} to ${outputJsonPath}...`);
      await unpack(testPresetPath, outputJsonPath);
      console.log("Conversion successful!");

      // Read and display the JSON structure
      const jsonData = JSON.parse(fs.readFileSync(outputJsonPath, "utf8"));
      console.log("\nPreset metadata:");
      console.log(JSON.stringify(jsonData.metadata, null, 2));

      console.log("\nData sections:");
      console.log(Object.keys(jsonData.data));
    } catch (error) {
      console.error("Conversion failed:", error.message);
      process.exit(1);
    }
  }

  testConversion();
} catch (error) {
  console.error("Failed to import node-serum2-preset-packager:", error.message);
  console.log("\nTrying alternative approach...");

  // Alternative: try to use the source files directly
  const packagePath = path.join(
    __dirname,
    "node_modules",
    "node-serum2-preset-packager"
  );
  console.log(`Package path: ${packagePath}`);

  if (fs.existsSync(packagePath)) {
    console.log("Package directory exists");
    const packageContents = fs.readdirSync(packagePath);
    console.log("Package contents:", packageContents);
  } else {
    console.log("Package directory not found");
  }

  process.exit(1);
}
