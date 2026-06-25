"""
download_repos.py — 克隆所有需要源代码仓库，构建训练数据语料库

从 GitHub 克隆 80+ 知名开源项目到 repos/ 目录，作为 prepare_data.py 的源码输入。
已存在的仓库会跳过（支持断点续传）。

用法:
  python3 scripts/download_repos.py                     # 克隆全部
  python3 scripts/download_repos.py --lang Swift        # 只克隆 Swift 仓库
  python3 scripts/download_repos.py --skip-linguist     # 跳过 GitHub Linguist
  python3 scripts/download_repos.py -j 8                # 8 个并行克隆
  python3 scripts/download_repos.py --shallow           # 浅克隆 (depth=1, 默认)
  python3 scripts/download_repos.py --full               # 完整克隆
"""

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ── 仓库配置 ─────────────────────────────────────────────────
# 格式: { "语言名": [("本地目录名", "git_url"), ...] }
# 所有 SSH URL 已转为 HTTPS 以方便未配置 SSH 的用户

REPOS = {
    "Swift": [
        ("Alamofire", "https://github.com/Alamofire/Alamofire.git"),
        ("RxSwift", "https://github.com/ReactiveX/RxSwift.git"),
        ("Kingfisher", "https://github.com/onevcat/Kingfisher.git"),
        ("SnapKit", "https://github.com/SnapKit/SnapKit.git"),
        ("lottie-ios", "https://github.com/airbnb/lottie-ios.git"),
        ("SwiftyJSON", "https://github.com/SwiftyJSON/SwiftyJSON.git"),
        ("SwiftUIX", "https://github.com/SwiftUIX/SwiftUIX.git"),
        ("vapor", "https://github.com/vapor/vapor.git"),
        ("swift-composable-architecture", "https://github.com/pointfreeco/swift-composable-architecture.git"),
        ("swift-snapshot-testing", "https://github.com/pointfreeco/swift-snapshot-testing.git"),
        ("swiftui-introspect", "https://github.com/siteline/swiftui-introspect.git"),
        ("KeyboardKit", "https://github.com/KeyboardKit/KeyboardKit.git"),
        ("Pulse", "https://github.com/kean/Pulse.git"),
        ("Highlightr", "https://github.com/raspu/Highlightr.git"),
        ("clean-architecture-swiftui", "https://github.com/nalexn/clean-architecture-swiftui.git"),
        ("ActivityIndicatorView", "https://github.com/exyte/ActivityIndicatorView.git"),
        ("AnimatedTabBar", "https://github.com/exyte/AnimatedTabBar.git"),
        ("Chat", "https://github.com/exyte/Chat.git"),
        ("ConcentricOnboarding", "https://github.com/exyte/ConcentricOnboarding.git"),
        ("Grid", "https://github.com/exyte/Grid.git"),
        ("PostApp", "https://github.com/Dimillian/PostApp.git"),
    ],
    "Objective-C": [
        ("AFNetworking", "https://github.com/AFNetworking/AFNetworking.git"),
        ("CocoaAsyncSocket", "https://github.com/robbiehanson/CocoaAsyncSocket.git"),
        ("CocoaLumberjack", "https://github.com/CocoaLumberjack/CocoaLumberjack.git"),
        ("FMDB", "https://github.com/ccgus/fmdb.git"),
        ("MBProgressHUD", "https://github.com/jdg/MBProgressHUD.git"),
        ("Mantle", "https://github.com/Mantle/Mantle.git"),
        ("Masonry", "https://github.com/Masonry/Masonry.git"),
        ("PureLayout", "https://github.com/PureLayout/PureLayout.git"),
        ("ReactiveObjC", "https://github.com/ReactiveCocoa/ReactiveObjC.git"),
        ("RestKit", "https://github.com/RestKit/RestKit.git"),
        ("SDWebImage", "https://github.com/SDWebImage/SDWebImage.git"),
        ("SVProgressHUD", "https://github.com/SVProgressHUD/SVProgressHUD.git"),
        ("Shimmer", "https://github.com/facebookarchive/Shimmer.git"),
        ("WordPress-iOS", "https://github.com/wordpress-mobile/WordPress-iOS.git"),
        ("facebook-ios-sdk", "https://github.com/facebook/facebook-ios-sdk.git"),
        ("image-crop-picker", "https://github.com/ivpusic/react-native-image-crop-picker.git"),
        ("realm", "https://github.com/realm/realm-swift.git"),
    ],
    "Kotlin": [
        ("kotlinx-coroutines", "https://github.com/Kotlin/kotlinx.coroutines.git"),
        ("Exposed", "https://github.com/JetBrains/Exposed.git"),
        ("ktor", "https://github.com/ktorio/ktor.git"),
        ("RxKotlin", "https://github.com/ReactiveX/RxKotlin.git"),
        ("architecture-samples", "https://github.com/android/architecture-components-samples.git"),
        ("okhttp", "https://github.com/square/okhttp.git"),
    ],
    "Java": [
        ("spring-framework", "https://github.com/spring-projects/spring-framework.git"),
        ("gson", "https://github.com/google/gson.git"),
        ("retrofit", "https://github.com/square/retrofit.git"),
    ],
    "JavaScript": [
        ("express", "https://github.com/expressjs/express.git"),
        ("lodash", "https://github.com/lodash/lodash.git"),
        ("react", "https://github.com/facebook/react.git"),
        ("vue", "https://github.com/vuejs/core.git"),
        ("axios", "https://github.com/axios/axios.git"),
        ("moment", "https://github.com/moment/moment.git"),
        ("redux", "https://github.com/reduxjs/redux.git"),
    ],
    "TypeScript": [
        ("DefinitelyTyped", "https://github.com/DefinitelyTyped/DefinitelyTyped.git"),
    ],
    "Python": [
        ("django", "https://github.com/django/django.git"),
        ("flask", "https://github.com/pallets/flask.git"),
        ("fastapi", "https://github.com/tiangolo/fastapi.git"),
        ("numpy", "https://github.com/numpy/numpy.git"),
        ("pandas", "https://github.com/pandas-dev/pandas.git"),
        ("requests", "https://github.com/psf/requests.git"),
        ("scrapy", "https://github.com/scrapy/scrapy.git"),
    ],
    "Go": [
        ("go", "https://github.com/golang/go.git"),
        ("grpc-go", "https://github.com/grpc/grpc-go.git"),
        ("cobra", "https://github.com/spf13/cobra.git"),
        ("gin", "https://github.com/gin-gonic/gin.git"),
    ],
    "Rust": [
        ("tokio", "https://github.com/tokio-rs/tokio.git"),
        ("ripgrep", "https://github.com/BurntSushi/ripgrep.git"),
        ("clap", "https://github.com/clap-rs/clap.git"),
        ("hyper", "https://github.com/hyperium/hyper.git"),
        ("rayon", "https://github.com/rayon-rs/rayon.git"),
        ("regex", "https://github.com/rust-lang/regex.git"),
        ("serde", "https://github.com/serde-rs/serde.git"),
    ],
    "C": [
        ("redis", "https://github.com/redis/redis.git"),
        ("curl", "https://github.com/curl/curl.git"),
        ("openssl", "https://github.com/openssl/openssl.git"),
    ],
    "C++": [
        ("protobuf", "https://github.com/protocolbuffers/protobuf.git"),
        ("abseil", "https://github.com/abseil/abseil-cpp.git"),
        ("googletest", "https://github.com/google/googletest.git"),
        ("nlohmann_json", "https://github.com/nlohmann/json.git"),
        ("fmt", "https://github.com/fmtlib/fmt.git"),
        ("spdlog", "https://github.com/gabime/spdlog.git"),
    ],
    "C#": [
        ("aspnetcore", "https://github.com/dotnet/aspnetcore.git"),
        ("NewtonsoftJson", "https://github.com/JamesNK/Newtonsoft.Json.git"),
        ("Dapper", "https://github.com/DapperLib/Dapper.git"),
    ],
    "Ruby": [
        ("rails", "https://github.com/rails/rails.git"),
        ("jekyll", "https://github.com/jekyll/jekyll.git"),
    ],
    "PHP": [
        ("laravel", "https://github.com/laravel/framework.git"),
        ("composer", "https://github.com/composer/composer.git"),
        ("monolog", "https://github.com/Seldaek/monolog.git"),
    ],
    "Shell": [
        ("ohmyzsh", "https://github.com/ohmyzsh/ohmyzsh.git"),
        ("nvm", "https://github.com/nvm-sh/nvm.git"),
        ("brew", "https://github.com/Homebrew/brew.git"),
        ("docker-ce", "https://github.com/docker/docker-ce.git"),
        ("docker-cli", "https://github.com/docker/cli.git"),
        ("git", "https://github.com/git/git.git"),
        ("fish-shell", "https://github.com/fish-shell/fish-shell.git"),
        ("antigen", "https://github.com/zsh-users/antigen.git"),
        ("asdf", "https://github.com/asdf-vm/asdf.git"),
        ("bash-it", "https://github.com/Bash-it/bash-it.git"),
        ("git-extras", "https://github.com/tj/git-extras.git"),
        ("pi-hole", "https://github.com/pi-hole/pi-hole.git"),
        ("powerline-fonts", "https://github.com/powerline/fonts.git"),
        ("pyenv", "https://github.com/pyenv/pyenv.git"),
        ("rbenv", "https://github.com/rbenv/rbenv.git"),
        ("rvm", "https://github.com/rvm/rvm.git"),
        ("sdkman", "https://github.com/sdkman/sdkman-cli.git"),
        ("starship", "https://github.com/starship/starship.git"),
        ("z", "https://github.com/rupa/z.git"),
        ("zsh-completions", "https://github.com/zsh-users/zsh-completions.git"),
    ],
    "CSS": [
        ("bootstrap", "https://github.com/twbs/bootstrap.git"),
        ("tailwindcss", "https://github.com/tailwindlabs/tailwindcss.git"),
        ("bulma", "https://github.com/jgthms/bulma.git"),
        ("foundation-sites", "https://github.com/foundation/foundation-sites.git"),
        ("ionic-framework", "https://github.com/ionic-team/ionic-framework.git"),
        ("materialize", "https://github.com/Dogfalo/materialize.git"),
        ("normalize.css", "https://github.com/necolas/normalize.css.git"),
        ("sass", "https://github.com/sass/sass.git"),
        ("postcss", "https://github.com/postcss/postcss.git"),
        ("pure", "https://github.com/pure-css/pure.git"),
        ("milligram", "https://github.com/milligram/milligram.git"),
        ("styled-components", "https://github.com/styled-components/styled-components.git"),
        ("my-mac-os", "https://github.com/nikitavoloboev/my-mac-os.git"),
    ],
    "HTML": [
        ("bulma", "https://github.com/jgthms/bulma.git"),
        ("tailwindcss", "https://github.com/tailwindlabs/tailwindcss.git"),
    ],
    "Dart": [
        ("sdk", "https://github.com/dart-lang/sdk.git"),
        ("samples", "https://github.com/flutter/samples.git"),
        ("labs", "https://github.com/dart-lang/labs.git"),
        ("Best-Flutter-UI-Templates", "https://github.com/mitesh77/Best-Flutter-UI-Templates.git"),
    ],
    "Lua": [
        ("neovim", "https://github.com/neovim/neovim.git"),
        ("mpv", "https://github.com/mpv-player/mpv.git"),
        ("lazy.nvim", "https://github.com/folke/lazy.nvim.git"),
        ("kickstart.nvim", "https://github.com/nvim-lua/kickstart.nvim.git"),
        ("telescope.nvim", "https://github.com/nvim-telescope/telescope.nvim.git"),
        ("nvim-cmp", "https://github.com/hrsh7th/nvim-cmp.git"),
        ("nvim-tree.lua", "https://github.com/nvim-tree/nvim-tree.lua.git"),
        ("plenary.nvim", "https://github.com/nvim-lua/plenary.nvim.git"),
        ("tokyonight.nvim", "https://github.com/folke/tokyonight.nvim.git"),
        ("luarocks", "https://github.com/luarocks/luarocks.git"),
        ("LuaSnip", "https://github.com/L3MON4D3/LuaSnip.git"),
        ("StyLua", "https://github.com/JohnnyMorganz/StyLua.git"),
        ("lua-nginx-module", "https://github.com/openresty/lua-nginx-module.git"),
    ],
    "SQL": [
        ("spark", "https://github.com/apache/spark.git"),
        ("hive", "https://github.com/apache/hive.git"),
        ("calcite", "https://github.com/apache/calcite.git"),
        ("derby", "https://github.com/apache/derby.git"),
        ("flyway", "https://github.com/flyway/flyway.git"),
        ("liquibase", "https://github.com/liquibase/liquibase.git"),
        ("postgres", "https://github.com/postgres/postgres.git"),
        ("sharding-jdbc", "https://github.com/dangdangdotcom/sharding-jdbc.git"),
        ("thingsboard", "https://github.com/thingsboard/thingsboard.git"),
    ],
    "XML": [
        ("spring-boot", "https://github.com/spring-projects/spring-boot.git"),
        ("tomcat", "https://github.com/apache/tomcat.git"),
        ("kubernetes", "https://github.com/kubernetes/kubernetes.git"),
    ],
    "YAML": [
        ("ansible", "https://github.com/ansible/ansible.git"),
    ],
    "JSON": [],   # 使用 Linguist samples 即可满足需求
}

LINGUIST_URL = "https://github.com/github/linguist.git"


# ── 工具函数 ─────────────────────────────────────────────────

def dir_size(path: str) -> str:
    """返回目录的近似大小（可读格式）。"""
    try:
        total = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                try:
                    total += os.path.getsize(fp)
                except OSError:
                    pass
        if total < 1024:
            return f"{total} B"
        elif total < 1024 * 1024:
            return f"{total / 1024:.1f} KB"
        else:
            return f"{total / (1024 * 1024):.1f} MB"
    except Exception:
        return "?"


def git_clone(url: str, target: str, shallow: bool = True) -> bool:
    """克隆单个 git 仓库。返回 True 表示成功，False 表示失败。"""
    if os.path.exists(target) and os.path.exists(os.path.join(target, ".git")):
        return True  # 已存在则跳过

    cmd = ["git", "clone"]
    if shallow:
        cmd.extend(["--depth", "1"])
    cmd.extend([url, target])

    result = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        timeout=600,
    )
    if result.returncode != 0:
        print(f"  ✗ 克隆失败: {url}")
        print(f"    错误: {result.stderr.decode('utf-8', errors='replace').strip()}")
        return False
    return True


def clone_repo(lang: str, name: str, url: str, repos_base: str, shallow: bool) -> tuple:
    """克隆单个仓库，返回 (lang, name, success, size_str)。"""
    lang_dir = os.path.join(repos_base, lang)
    os.makedirs(lang_dir, exist_ok=True)
    target = os.path.join(lang_dir, name)

    short = f"{lang}/{name}"
    if os.path.exists(target) and os.path.exists(os.path.join(target, ".git")):
        size = dir_size(target)
        return (lang, name, True, size, "SKIP")

    print(f"  克隆 {short} ...", end=" ", flush=True)
    t0 = time.time()
    ok = git_clone(url, target, shallow=shallow)
    elapsed = time.time() - t0
    if ok:
        size = dir_size(target)
        print(f"完成 ({elapsed:.1f}s, {size})")
        return (lang, name, True, size, "CLONE")
    else:
        print(f"失败 ({elapsed:.1f}s)")
        return (lang, name, False, "?", "FAIL")


# ── 主流程 ───────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="下载训练数据所需的源代码仓库"
    )
    parser.add_argument(
        "--lang", "-l",
        help="只下载指定语言的仓库（如 --lang Swift）",
    )
    parser.add_argument(
        "--skip-linguist", action="store_true",
        help="跳过克隆 GitHub Linguist",
    )
    parser.add_argument(
        "--shallow", action="store_true", default=True,
        help="使用浅克隆 depth=1（默认）",
    )
    parser.add_argument(
        "--full", action="store_false", dest="shallow",
        help="完整克隆（包含完整 git 历史）",
    )
    parser.add_argument(
        "--jobs", "-j", type=int, default=4,
        help="并行克隆数（默认: 4）",
    )
    args = parser.parse_args()

    # 确定项目根目录 (scripts/../repos)
    repos_base = str(Path(__file__).resolve().parent.parent / "repos")
    os.makedirs(repos_base, exist_ok=True)

    print("=" * 60)
    print("CodeLaunguageClassifier — 下载源代码仓库")
    print("=" * 60)
    print(f"目标目录: {repos_base}")
    print(f"浅克隆: {'是' if args.shallow else '否'}")
    print(f"并行数: {args.jobs}")
    print()

    # ── 1. 克隆 GitHub Linguist ──
    if not args.skip_linguist:
        linguist_dir = os.path.join(repos_base, "linguist")
        print(f"[1/2] GitHub Linguist")
        print(f"  URL: {LINGUIST_URL}")
        if os.path.exists(linguist_dir) and os.path.exists(os.path.join(linguist_dir, ".git")):
            size = dir_size(linguist_dir)
            print(f"  已存在 ({size})")
        else:
            print(f"  克隆中 ...", end=" ", flush=True)
            t0 = time.time()
            ok = git_clone(LINGUIST_URL, linguist_dir, shallow=args.shallow)
            elapsed = time.time() - t0
            if ok:
                size = dir_size(linguist_dir)
                print(f"完成 ({elapsed:.1f}s, {size})")
            else:
                print("失败")
        print()

    # ── 2. 克隆各语言仓库 ──
    print(f"[2/2] 各语言项目仓库")
    languages = [args.lang] if args.lang else list(REPOS.keys())

    all_tasks = []
    for lang in languages:
        if lang not in REPOS:
            print(f"  警告: 未知语言 '{lang}'，跳过")
            continue
        for name, url in REPOS[lang]:
            all_tasks.append((lang, name, url))

    if not all_tasks:
        print("  （没有需要下载的仓库）")
        return

    total = len(all_tasks)
    print(f"  共 {total} 个仓库")
    print()

    results = []
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(clone_repo, lang, name, url, repos_base, args.shallow): (lang, name)
            for lang, name, url in all_tasks
        }
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                lang, name = futures[future]
                results.append((lang, name, False, "?", f"ERROR: {e}"))

    # ── 汇总 ──
    print()
    print("=" * 60)
    print("下载汇总")
    print("=" * 60)

    by_lang = {}
    for lang, name, ok, size, status in results:
        by_lang.setdefault(lang, []).append((name, ok, size, status))

    total_ok = sum(1 for _, _, ok, _, _ in results if ok)
    total_fail = sum(1 for _, _, ok, _, _ in results if not ok)

    for lang in sorted(by_lang.keys()):
        repos = by_lang[lang]
        ok_count = sum(1 for _, ok, _, _ in repos if ok)
        fail_count = sum(1 for _, ok, _, _ in repos if not ok)
        total_size = sum(
            parse_size(size) for _, ok, size, _ in repos if ok and size != "?"
        )
        if total_size > 0:
            if total_size < 1024:
                size_str = f"{total_size} B"
            elif total_size < 1024 * 1024:
                size_str = f"{total_size / 1024:.0f} KB"
            else:
                size_str = f"{total_size / (1024 * 1024):.1f} MB"
        else:
            size_str = "?"
        status_icon = "✓" if fail_count == 0 else f"✓{ok_count} ✗{fail_count}"
        print(f"  {lang:<15} {status_icon}  ({len(repos)} 仓库, 约 {size_str})")

    print("-" * 40)
    print(f"  总数: {total_ok}/{total} 成功")
    if total_fail > 0:
        print(f"  失败: {total_fail} 个（可重试）")


def parse_size(size_str: str) -> int:
    """将可读大小字符串转换为字节数。"""
    if not size_str or size_str == "?":
        return 0
    parts = size_str.split()
    if len(parts) != 2:
        return 0
    try:
        val = float(parts[0])
        unit = parts[1]
        if unit == "KB":
            return int(val * 1024)
        elif unit == "MB":
            return int(val * 1024 * 1024)
        elif unit == "B":
            return int(val)
        else:
            return 0
    except ValueError:
        return 0


if __name__ == "__main__":
    main()
