#!/usr/bin/env bash
# Runs inside GitHub Actions after the ZIP was unpacked. Do not run locally.
set -euo pipefail
cd "$(dirname "$0")/.."
ZIPNAME="${1:-zip}"

read_json() { python3 -c "import json;print(json.load(open('docs/version.json')).get('$1','$2'))"; }
VERSION=$(read_json version 1.0.0)
CODE=$((1000 + ${GITHUB_RUN_NUMBER:-0}))
NOTES=$(read_json notes "")
OWNER="${GITHUB_REPOSITORY%%/*}"
REPO="${GITHUB_REPOSITORY##*/}"
echo "== Al-Ansar $VERSION (build $CODE) from $ZIPNAME =="

# 1) app config (repo name for in-app update checks) + Colab notebook personalisation
echo "{\"repo\": \"$GITHUB_REPOSITORY\"}" > docs/app-config.json
python3 - "$OWNER" "$REPO" << 'PY'
import sys, glob
o, r = sys.argv[1:3]
for p in glob.glob("colab/*.ipynb"):
    s = open(p, encoding="utf-8").read()
    s = s.replace("YOUR_GITHUB_USERNAME", o).replace('REPO = \\"AlAnsar-Voice-Cloning\\"', 'REPO = \\"%s\\"' % r)
    open(p, "w", encoding="utf-8").write(s)
PY

# 2) fixed signing key, so every new APK can update the installed one
mkdir -p signing "$HOME/.android"
if [ ! -f signing/debug.keystore ]; then
  keytool -genkey -v -keystore signing/debug.keystore -storepass android -alias androiddebugkey \
    -keypass android -keyalg RSA -keysize 2048 -validity 10000 \
    -dname "CN=Android Debug,O=Android,C=US"
fi
[ -f signing/debug.keystore ] || { echo "ERROR: signing key could not be created"; exit 1; }
cp signing/debug.keystore "$HOME/.android/debug.keystore"
echo "Signing key fingerprint (must stay the same in every build):"
keytool -list -v -keystore signing/debug.keystore -storepass android -alias androiddebugkey 2>/dev/null | grep -i "SHA256" || true

# 3) save the unpacked source back to the repository
git config user.name "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
git add -A
git diff --cached --quiet || git commit -q -m "Al-Ansar $VERSION: unpacked $ZIPNAME"
git push origin HEAD:"$GITHUB_REF_NAME"

# 4) try to switch on the web app (GitHub Pages); harmless if it is already on or not allowed
gh api -X POST "repos/$GITHUB_REPOSITORY/pages" -f "source[branch]=$GITHUB_REF_NAME" -f "source[path]=/docs" > /dev/null 2>&1 || true

# 5) build the Android APK
cd mobile
rm -rf www android
mkdir -p www
cp -r ../docs/* www/
rm -f www/sw.js
npm install --no-audit --no-fund
npx cap add android
MANIFEST=android/app/src/main/AndroidManifest.xml
sed -i 's#<uses-permission android:name="android.permission.INTERNET" />#<uses-permission android:name="android.permission.INTERNET" />\n    <uses-permission android:name="android.permission.RECORD_AUDIO" />\n    <uses-permission android:name="android.permission.MODIFY_AUDIO_SETTINGS" />#' "$MANIFEST"
sed -i "s/versionCode 1/versionCode $CODE/; s/versionName \"1.0\"/versionName \"$VERSION\"/" android/app/build.gradle
# in-app installer (native plugin); if it fails to build, rebuild without it so a release still ships
PKG=$(python3 -c "import json;print(json.load(open('capacitor.config.json'))['appId'])")
PKGDIR="$PWD/android/app/src/main/java/$(echo "$PKG" | tr . /)"
cp "$PKGDIR/MainActivity.java" /tmp/MainActivity.orig
if [ -f native/ApkInstallerPlugin.java ]; then
  sed "s/__PKG__/$PKG/g" native/ApkInstallerPlugin.java > "$PKGDIR/ApkInstallerPlugin.java"
  cat > "$PKGDIR/MainActivity.java" << JAVA
package $PKG;

import android.os.Bundle;
import com.getcapacitor.BridgeActivity;

public class MainActivity extends BridgeActivity {
    @Override
    public void onCreate(Bundle savedInstanceState) {
        registerPlugin(ApkInstallerPlugin.class);
        super.onCreate(savedInstanceState);
    }
}
JAVA
  sed -i 's#<uses-permission android:name="android.permission.INTERNET" />#<uses-permission android:name="android.permission.INTERNET" />\n    <uses-permission android:name="android.permission.REQUEST_INSTALL_PACKAGES" />#' "$MANIFEST"
fi
npx cap sync android
cd android
chmod +x gradlew
if ! ./gradlew assembleDebug --no-daemon; then
  echo "::warning::in-app installer failed to build; rebuilding without it"
  cp /tmp/MainActivity.orig "$PKGDIR/MainActivity.java"
  rm -f "$PKGDIR/ApkInstallerPlugin.java"
  ./gradlew assembleDebug --no-daemon
fi
cp app/build/outputs/apk/debug/app-debug.apk "$GITHUB_WORKSPACE/AlAnsar-VoiceCloning.apk"
cd "$GITHUB_WORKSPACE"

# 6) publish as a GitHub Release (stable link: releases/latest/download/AlAnsar-VoiceCloning.apk)
gh release delete "v$VERSION" --cleanup-tag -y > /dev/null 2>&1 || true
gh release create "v$VERSION" AlAnsar-VoiceCloning.apk \
  --title "Al-Ansar Voice Cloning v$VERSION" --notes "$NOTES" --latest
echo "== Done: https://github.com/$GITHUB_REPOSITORY/releases/latest =="
